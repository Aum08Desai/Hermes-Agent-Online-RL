# Hermes Agent Online RL

Online reinforcement learning for [Hermes Agent](https://github.com/NousResearch/hermes-agent), with local MIS-PO training and a Tinker-backed SDPO-style path for hosted models.

Each Hermes response can be treated as a training trajectory. This project adds an online update loop around that trajectory: after each response, you can record a **binary reward** (upweight / downweight / skip) or, on **Tinker**, run an **SDPO-style** mode that pairs the positive or negative label with a short text note. The local path uses a scalar MIS-PO-style update. The Tinker SDPO path uses a frozen-teacher, top-K distillation approximation based on [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) and the reference implementation in [lasgroup/SDPO](https://github.com/lasgroup/SDPO).

---

## Why Online RL on Hermes Agent?

Traditional RLHF requires collecting large datasets, training offline, and deploying new checkpoints. Online RL collapses this into a single continuous loop:

| Traditional RLHF | Hermes Online RL |
|---|---|
| Collect thousands of preference pairs | Rate responses as you use them |
| Train offline on a GPU cluster | LoRA trains locally in the background |
| Deploy a new model checkpoint | Adapter hot-loads into your running server |
| Shared offline checkpoint | Feedback collected during live usage |
| Weeks of iteration | Minutes between feedback and improvement |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hermes Agent CLI                          │
│                                                             │
│  User prompt ──► Local Model (vLLM/Ollama/MLX) ──► Response │
│                         │                          │        │
│                    LoRA Adapter                     │        │
│                    (hot-loaded)              ┌──────┴──────┐ │
│                         ▲                   │  Feedback UI │ │
│                         │                   │  [1] ⬆ Up    │ │
│                         │                   │  [2] ⬇ Down  │ │
│                         │                   │  [3] ⊘ Skip  │ │
│                         │                   └──────┬──────┘ │
│                         │                          │        │
│                   ┌─────┴──────┐                   │        │
│                   │  MIS-PO    │◄──────────────────┘        │
│                   │  Trainer   │                             │
│                   │ PyTorch/MLX│  Fires after min_batch_size │
│                   └────────────┘  feedback samples (def: 8)  │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Backend Training

The trainer automatically selects the best backend for your hardware:

| Hardware | Backend | How It Works |
|---|---|---|
| **Any machine with API access** | Tinker | Hosted online RL with Tinker-managed training and inference. Supports the binary path and a frozen-teacher SDPO-style path. |
| **Apple Silicon (M1–M5)** | MLX | Native Metal acceleration via `mlx` + `mlx-lm`. Zero-copy unified memory. |
| **NVIDIA GPU** | PyTorch (CUDA) | Standard PyTorch with `torch.cuda`. BF16/FP16 auto-selected. |
| **Apple (no MLX installed)** | PyTorch (MPS) | Falls back to Metal Performance Shaders via PyTorch. |
| **CPU** | PyTorch (CPU) | Works everywhere, just slower. FP32. |

**Auto-detection is the default.** Set `device: auto` (or omit it) and the trainer picks the right backend. You can override with `device: mlx`, `device: cuda`, `device: mps`, or `device: cpu`.

### The Feedback Loop

1. **You chat with Hermes** using any workflow — coding, research, writing, tool use
2. **After each response**, a compact feedback panel appears with three options:
   - **Upweight** (`1` or `↑/Enter`) — reward = +1.0, "learn from this"
   - **Downweight** (`2` or `↓/Enter`) — reward = -1.0, "unlearn this"
   - **Skip** (`3` or `↓↓/Enter`) — no training signal
   - In **SDPO mode**, Hermes follows the up/down label with a short text note prompt so the teacher can distill from richer feedback.
3. **Feedback accumulates** in a local SQLite database with full trajectory context
4. **When `min_batch_size` samples are collected** (default: 8), the MIS-PO trainer fires automatically in the background
5. **The trained LoRA adapter is hot-loaded** into your running vLLM or Ollama server (or saved to disk for MLX)
6. **Next response uses the updated adapter** — the loop continues

---

## MIS-PO: The RL Algorithm

This implementation uses **Metropolis Independence Sampling - Filtered Policy Optimization (MIS-PO)**, the same algorithm introduced in the [Step 3.5 Flash technical report](https://arxiv.org/abs/2602.10604). MIS-PO is specifically designed for stable RL on language models without requiring grouped rollouts.

### Why MIS-PO over PPO/GRPO?

| Algorithm | Requires | Issue for Online RL |
|---|---|---|
| PPO | Grouped rollouts, critic network | Too heavy — needs multiple responses per prompt, doubles VRAM for critic |
| GRPO | Multiple rollouts per prompt | Can't do this in real-time — user only sees one response |
| REINFORCE | Nothing extra | High variance, unstable |
| **MIS-PO** | **Single trajectory + binary filtering** | **Perfect for online RL — one response = one trajectory** |

### The Objective

MIS-PO uses a REINFORCE-style policy gradient with binary masking instead of continuous importance weighting:

```
L_actor = -E[I(τ) · log π_θ(a_t|s_t) · Â_t]
```

Where `I(τ)` is a binary indicator that filters off-policy samples at two granularities:

**Token-level filtering:** For each token, compute the probability ratio between the training policy and the inference policy:

```
x_t = π_θ_train(a_t|s_t) / π_θ_inference(a_t|s_t)
```

Only tokens where `ρ_min ≤ x_t ≤ ρ_max` contribute to the gradient. Default bounds: `[0.8, 1.25]`.

**Trajectory-level filtering:** Compute the geometric mean of token ratios across the full assistant response:

```
ρ̄(τ) = (∏_t x_t)^(1/T)
```

The entire trajectory is rejected if `ρ̄(τ)` falls outside `[0.9, 1.1]`. This prevents accumulated small token-level drifts from corrupting the gradient.

### KL Regularization

An unbiased KL divergence penalty prevents the adapter from drifting too far from the base model:

```
L_KL = E[exp(log_ratio) - log_ratio - 1]
```

With coefficient `β = 0.001` (matching Step 3.5 Flash). This is the Schulman k3 estimator — unbiased and low-variance.

**VRAM impact of KL: zero.** The KL term reuses tensors already computed for the actor loss (`new_logprobs`, `old_logprobs`). No additional forward pass or model copy needed.

### Why Binary Filtering > Clipping

PPO clips the importance ratio to `[1-ε, 1+ε]`, which still passes scaled gradients from off-policy tokens. MIS-PO completely excludes off-policy tokens, treating the remaining ones as on-policy. This:

- Reduces gradient variance (empirically 5x lower than PPO per Step 3.5 Flash)
- Is more stable for long trajectories where small per-token errors compound
- Works with a single trajectory (no need for the "group" in GRPO)

---

## SDPO: The Self-Distillation Path

The `sdpo` mode is a Tinker-backed approximation of **Self-Distillation Policy Optimization (SDPO)** from [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802), based on the public design in the paper and the reference implementation in [lasgroup/SDPO](https://github.com/lasgroup/SDPO).

### Why SDPO for Hermes?

The paper's main result is that dense distillation targets extracted from the model's own successful behavior can improve online RL relative to scalar-only updates. In Hermes, that matters when feedback includes more than a binary label:

- The **label** (`upweight` / `downweight`) provides the optimization direction
- The **text note** provides a dense correction or preservation signal
- The **teacher distribution** provides token-level targets beyond the sampled trajectory

This matches continual agentic usage better than grouped-rollout methods because Hermes still trains on one accepted response at a time.

### Hermes SDPO Approximation

The current implementation is intentionally constrained by Tinker's public training API:

- Teacher mode defaults to `frozen`
- Distillation uses top-K teacher targets controlled by `sdpo_topk`
- The SDPO loss is combined with the existing scalar online RL signal
- `train_steps: 0` means one pass over newly collected feedback instead of a fixed offline training schedule
- LoRA defaults are separate from the binary / MIS-PO defaults

This keeps the update loop compatible with continual online usage while preserving the main SDPO structure that Hermes can express through Tinker.

### Objective Shape

At a high level, the Hermes SDPO path optimizes a weighted combination of:

```text
L_total = λ_rl * L_binary_rl + λ_distill * L_teacher_distill
```

Where:

- `L_binary_rl` is the existing scalar online RL term driven by the positive or negative feedback label
- `L_teacher_distill` matches the student policy to frozen teacher targets over the response tokens using top-K teacher outputs
- `λ_rl` and `λ_distill` correspond to `sdpo_rl_weight` and `sdpo_distillation_weight`

### Differences from the Paper

Hermes does not currently implement the full SDPO training stack from the paper:

- no EMA or trust-region teacher updates
- no full-logit plus tail distillation objective
- no full `verl` / Ray training topology

The implemented version is a frozen-teacher, top-K distillation approximation designed for Tinker's hosted training interface and Hermes' single-trajectory online feedback loop.

---

## LoRA Training Details

### Why LoRA?

Full fine-tuning would require storing optimizer states for every parameter — 2-4x the model's VRAM. LoRA trains only low-rank adapter matrices (~0.1-1% of parameters), keeping VRAM overhead minimal:

| Component | VRAM |
|---|---|
| Base model (frozen) | Same as inference |
| LoRA adapter parameters | ~10-50 MB |
| LoRA gradients + optimizer states | ~30-150 MB |
| One forward/backward pass activations | Same as inference |
| **Total training overhead** | **~50-200 MB above inference** |

Since you're already running the model locally for inference, training adds minimal extra VRAM.

### MLX vs PyTorch LoRA

Both backends implement identical LoRA training with the same MIS-PO objective. The difference is the underlying framework:

| Aspect | PyTorch Backend | MLX Backend |
|---|---|---|
| Library | `torch` + `peft` | `mlx` + `mlx-lm` |
| Adapter format | PEFT safetensors | mlx-lm `adapters.safetensors` |
| LoRA application | `peft.get_peft_model()` | `mlx_lm.tuner.linear_to_lora_layers()` |
| Autodiff | `loss.backward()` | `mlx.nn.value_and_grad()` |
| Optimizer | `torch.optim.AdamW` | `mlx.optimizers.AdamW` |
| Memory model | Separate CPU/GPU memory | Unified memory (zero-copy) |
| Quantized training | Manual setup | Automatic QLoRA when loading 4-bit models |

**Apple Silicon advantage:** MLX leverages unified memory, meaning the model, optimizer states, and activations share the same memory pool — no CPU↔GPU copies. A 7B model that needs 14GB for inference might only need ~14.5GB total for inference + LoRA training.

### Default Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| LoRA rank | 16 | Good balance of capacity vs. VRAM |
| LoRA alpha | 32 | 2x rank (standard scaling) |
| LoRA dropout | 0.05 | Light regularization |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All attention + MLP projections |
| Learning rate | 2e-6 | Conservative for stability |
| Weight decay | 0.1 | AdamW regularization |
| Warmup steps | 20 | Gradual LR ramp |
| Train steps per batch | 16 | Steps per training trigger |
| Gradient accumulation | 4 | Effective batch = 4 examples |
| Max sequence length | 262144 (256K) | Truncates longer trajectories |
| Grad norm clip | 1.0 | Prevents exploding gradients |
| Min batch size | 8 | Feedback samples before training fires |
| Max saved adapters | 4 | Keeps last 4, auto-cleans older ones |

### Adapter Lifecycle

1. **First training:** Creates a fresh LoRA adapter from the base model
2. **Subsequent trainings:** Loads the previous adapter and continues training (warm-start)
3. **Hot-loading:** After training, the adapter is published to vLLM (`/v1/load_lora_adapter`), Ollama (`ollama create`), or saved to disk (MLX)
4. **Cleanup:** Only the last `max_saved_adapters` checkpoints are kept on disk

---

## Installation & Setup

### Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed and working
- A local model server (vLLM, Ollama, llama.cpp, LM Studio) — or MLX for Apple Silicon native inference
- Python 3.11+

### Install

**For NVIDIA GPU / CPU (PyTorch backend):**

```bash
pip install -e ".[online-rl]"
```

This installs `torch`, `transformers`, and `peft`.

**For Apple Silicon (MLX backend):**

```bash
pip install -e ".[online-rl-mlx]"
```

This installs `mlx` and `mlx-lm`. The trainer auto-detects Apple Silicon and uses MLX natively.

**For both backends (maximum compatibility):**

```bash
pip install -e ".[online-rl,online-rl-mlx]"
```

**For hosted Tinker training and inference:**

```bash
pip install -e ".[online-rl-tinker]"
```

### Configure

```bash
# Interactive setup wizard
hermes setup online-rl

# Or configure manually
hermes config set online_rl.enabled true
hermes config set online_rl.prompt_after_response true
hermes config set online_rl.training_base_model "your-model-name"
hermes config set online_rl.local_only true
```

**MLX-specific example:**

```yaml
# ~/.hermes/config.yaml
online_rl:
  enabled: true
  prompt_after_response: true
  training_base_model: mlx-community/Mistral-7B-Instruct-v0.3-4bit
  device: auto  # auto-detects MLX on Apple Silicon
```

> **Tip:** Use quantized models from the [mlx-community](https://huggingface.co/mlx-community) on Hugging Face for best MLX performance. 4-bit models automatically get QLoRA — the base stays quantized while only LoRA weights train in full precision.

The setup wizard will:
1. Scan for local inference servers (vLLM, Ollama, llama.cpp, LM Studio)
2. Auto-detect available models
3. Let you configure LoRA hyperparameters
4. Write everything to `~/.hermes/config.yaml`

When you choose the Tinker backend, Hermes also reconfigures the main chat runtime to use Tinker's OpenAI-compatible inference endpoint with the same selected model.

### Use

```bash
hermes  # start chatting — feedback panel appears after each response
```

Press `1` to reinforce, `2` to correct, `3` to skip.

- In `binary` mode, `1` and `2` store scalar positive/negative feedback.
- In `sdpo` mode, `1` and `2` still set the direction, and Hermes then asks for a short text note.
- In SDPO mode, the button provides the polarity and the note provides the rich feedback content.

Training fires automatically once enough rated samples accumulate.

### Monitor

```bash
hermes online-rl status          # Show config and active adapter
hermes online-rl list-feedback   # List stored feedback
hermes online-rl list-feedback --pending  # Only untrained samples
```

---

## UI

### Feedback Panel

After each response, a compact panel appears:

```
╭─ Online RL ──────────────── 42s ─╮
│                                   │
│  ▲/▼ navigate  Enter confirm     │
│                                   │
│  ❯ ⬆ Upweight   learn from this  │
│    ⬇ Downweight  unlearn this     │
│    ⊘ Skip        no training      │
│                                   │
╰──────────────────────────────────╯
```

- **Countdown timer** in the title bar (auto-skips after timeout)
- **Number keys** `1`/`2`/`3` for instant selection
- **Arrow keys + Enter** for navigation
- **Ctrl+C** to skip

In SDPO mode, confirming `Upweight` or `Downweight` opens a short free-text prompt:

- `Upweight + note` means: this answer was good, reinforce it, and preserve the traits described in the note
- `Downweight + note` means: this answer was bad, correct it, and use the note as the failure/correction signal

### Status Bar

When online RL is active, a green `RL` indicator appears in the bottom status bar alongside the model name and context usage.

---

## Configuration Reference

All settings live under `online_rl:` in `~/.hermes/config.yaml`:

```yaml
online_rl:
  enabled: true
  prompt_after_response: true
  local_only: true                    # Only activate for localhost servers
  backend: auto                       # auto | tinker | vllm | ollama | mlx
  algorithm: binary                   # binary | sdpo
  training_base_model: "your-model"   # HuggingFace model name or path

  # Binary / MIS-PO-style settings
  token_ratio_min: 0.8
  token_ratio_max: 1.25
  trajectory_ratio_min: 0.9
  trajectory_ratio_max: 1.1
  kl_coefficient: 0.001

  # LoRA
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

  # Training
  learning_rate: 0.000002
  weight_decay: 0.1
  warmup_steps: 20
  train_steps: 16
  gradient_accumulation_steps: 4
  min_batch_size: 8
  max_batch_size: 64
  max_sequence_length: 262144

  # Adapter management
  adapter_name: hermes-online-rl
  max_saved_adapters: 4
  builtin_trainer: true

  # Tinker / SDPO settings
  tinker_base_model: ""
  tinker_lora_rank: 32
  tinker_loss_fn: importance_sampling
  sdpo_teacher_mode: frozen
  sdpo_topk: 20
  sdpo_rl_weight: 1.0
  sdpo_distillation_weight: 1.0
  sdpo_text_feedback_timeout_seconds: 90

  # Device — auto-detects MLX on Apple Silicon, CUDA on NVIDIA, etc.
  device: auto          # auto, mlx, cuda, mps, cpu
  torch_dtype: auto     # auto, bf16, fp16, fp32 (PyTorch backend only)
```

Environment variable overrides are available for all settings (e.g., `HERMES_ONLINE_RL_ENABLED=true`).

Hermes now keeps separate default profiles for the two modes:

- `binary` keeps the original scalar-reward online RL defaults
- `sdpo` uses its own Tinker-oriented defaults, including `learning_rate=1e-6`, `tinker_lora_rank=32`, `sdpo_topk=20`, and `train_steps=0` meaning one pass over newly collected feedback

---

## Files

| File | Purpose |
|---|---|
| `agent/online_rl.py` | Feedback capture, backend detection, adapter publishing (vLLM/Ollama/MLX/Tinker) |
| `agent/online_rl_trainer.py` | PyTorch binary/MIS-PO-style trainer with backend routing |
| `agent/online_rl_trainer_mlx.py` | MLX-native binary/MIS-PO-style trainer for Apple Silicon |
| `agent/online_rl_trainer_tinker.py` | Tinker API backend for hosted binary RL and SDPO-style training |
| `cli.py` | Feedback panel UI with icons, countdown, hotkeys, RL status bar |
| `hermes_cli/setup.py` | `setup_online_rl()` wizard with local and Tinker runtime configuration |
| `hermes_cli/main.py` | `hermes online-rl` subcommands |
| `hermes_state.py` | RL feedback SQLite schema and queries |
| `pyproject.toml` | `[online-rl]`, `[online-rl-mlx]`, and `[online-rl-tinker]` optional dependency groups |

---

## How It Works End-to-End

1. You run `hermes` and chat normally
2. Hermes generates a response using either your local runtime or Tinker's OpenAI-compatible inference runtime
3. The feedback panel appears — you press `1` (good), `2` (bad), or `3` (skip)
4. Feedback is stored in SQLite with the full message context (prompt + response)
5. In `sdpo` mode, Hermes also stores your short text note alongside the positive/negative label
6. When enough rated samples accumulate, `maybe_trigger_online_rl_training()` fires
7. The trainer spawns as a background subprocess:
   - **Auto-selects backend:** Tinker when `TINKER_API_KEY` + `tinker_base_model` are configured, otherwise MLX on Apple Silicon, PyTorch+CUDA on NVIDIA, PyTorch+MPS/CPU elsewhere
   - Loads the base model + existing LoRA adapter (if any)
   - In `binary` mode, runs scalar-reward online RL updates
   - In `sdpo` mode, runs a frozen-teacher SDPO approximation on Tinker with top-K distillation plus the scalar RL signal
   - Saves the updated adapter to `~/.hermes/online_rl/adapters/` or publishes the Tinker checkpoint metadata
   - Hot-loads it into the running inference server, saves it for MLX, or updates the active Tinker checkpoint/runtime metadata
8. Your next conversation uses the improved adapter or Tinker checkpoint
9. The cycle repeats — your model continuously improves at your specific workflows

---

## References

- [Step 3.5 Flash Technical Report](https://arxiv.org/abs/2602.10604) — MIS-PO algorithm
- [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) — SDPO paper
- [SDPO Reference Implementation](https://github.com/lasgroup/SDPO)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Base agent framework
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Trust Region Masking](https://arxiv.org/abs/2512.23075) — Theoretical basis for sequence-level trust regions
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM fine-tuning and inference on MLX

---

## License

MIT — same as Hermes Agent.
