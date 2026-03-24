"""MLX-native MIS-PO trainer for online LoRA updates on Apple Silicon."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner import linear_to_lora_layers

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from agent.online_rl import (
    load_online_rl_config,
    load_online_rl_state,
    publish_online_rl_adapter,
)


def is_mlx_available() -> bool:
    """Return True when MLX and mlx-lm are importable."""
    return MLX_AVAILABLE


@dataclass
class TrajectoryExample:
    feedback_id: int
    reward: float
    input_ids: List[int]
    assistant_start: int
    old_logprobs: Optional[List[float]] = None

    @property
    def target_token_count(self) -> int:
        return max(0, len(self.input_ids) - max(1, self.assistant_start))


# ---------------------------------------------------------------------------
# Text helpers (shared logic with the PyTorch trainer)
# ---------------------------------------------------------------------------

def _normalize_messages(messages) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages or []:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in {"text", "input_text", "output_text"}:
                        parts.append(str(item.get("text") or ""))
                elif item is not None:
                    parts.append(str(item))
            content = "\n".join(part for part in parts if part)
        normalized.append({"role": role, "content": str(content)})
    return normalized


def _fallback_chat_text(messages: List[Dict[str, str]], *, add_generation_prompt: bool) -> str:
    lines: List[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        lines.append(f"{role}: {message.get('content', '')}")
    if add_generation_prompt:
        lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _chat_text(tokenizer, messages: List[Dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    except Exception:
        pass
    return _fallback_chat_text(messages, add_generation_prompt=add_generation_prompt)


def _build_examples(export_rows: List[Dict[str, Any]], tokenizer, cfg: Dict[str, Any]) -> List[TrajectoryExample]:
    max_length = max(512, int(cfg.get("max_sequence_length", 4096)))
    examples: List[TrajectoryExample] = []
    for row in export_rows:
        reward = float(row.get("reward") or 0.0)
        if reward == 0:
            continue

        prompt_messages = _normalize_messages(row.get("prompt_messages") or [])
        assistant_message = row.get("assistant_message") or {}
        assistant_content = str(assistant_message.get("content") or row.get("response_text") or "").strip()
        if not assistant_content:
            continue

        full_messages = list(prompt_messages)
        full_messages.append({"role": "assistant", "content": assistant_content})

        prompt_text = _chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _chat_text(tokenizer, full_messages, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        assistant_start = len(prompt_ids)
        if len(full_ids) <= assistant_start:
            continue

        if len(full_ids) > max_length:
            trim = len(full_ids) - max_length
            full_ids = full_ids[trim:]
            assistant_start = max(0, assistant_start - trim)
            if assistant_start >= len(full_ids):
                continue

        examples.append(
            TrajectoryExample(
                feedback_id=int(row["feedback_id"]),
                reward=reward,
                input_ids=list(full_ids),
                assistant_start=int(assistant_start),
            )
        )
    return examples


# ---------------------------------------------------------------------------
# MLX model loading with LoRA
# ---------------------------------------------------------------------------

def _lora_config_from_cfg(cfg: Dict[str, Any]) -> dict:
    """Build an mlx-lm compatible LoRA config dict from our online-RL config."""
    rank = int(cfg.get("lora_rank", 16))
    alpha = int(cfg.get("lora_alpha", 32))
    dropout = float(cfg.get("lora_dropout", 0.05))
    # mlx-lm uses scale = alpha / rank
    scale = float(alpha) / float(rank) if rank > 0 else 1.0
    return {
        "num_layers": -1,  # -1 means apply to all layers
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": dropout,
        },
    }


def _load_training_model(cfg: Dict[str, Any]):
    """Load an MLX model with LoRA layers ready for training."""
    runtime_state = load_online_rl_state(cfg)
    base_model_name = str(
        runtime_state.get("base_model_name") or cfg.get("training_base_model") or ""
    ).strip()
    if not base_model_name:
        raise RuntimeError(
            "online_rl.training_base_model is required for MLX LoRA training"
        )

    # Check if there's an existing MLX adapter to resume from
    active_adapter_path = str(runtime_state.get("active_adapter_path") or "").strip()
    mlx_adapter_path = None
    if active_adapter_path:
        # Look for MLX adapter format (adapters.safetensors)
        mlx_adapter_file = Path(active_adapter_path) / "adapters.safetensors"
        if mlx_adapter_file.exists():
            mlx_adapter_path = str(Path(active_adapter_path))

    # Load model + tokenizer via mlx-lm
    model, tokenizer = mlx_load(
        base_model_name,
        adapter_path=mlx_adapter_path,
    )

    # Apply LoRA layers if not resuming from an adapter
    lora_cfg = _lora_config_from_cfg(cfg)
    if mlx_adapter_path is None:
        model.freeze()
        linear_to_lora_layers(
            model,
            lora_cfg["num_layers"],
            lora_cfg["lora_parameters"],
        )

    model.train()
    return model, tokenizer, base_model_name, lora_cfg


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def _compute_logprobs(model, input_ids: mx.array) -> mx.array:
    """Forward pass → per-token log-probabilities for the shifted sequence."""
    logits = model(input_ids[None, :])  # (1, seq_len, vocab)
    logits = logits[:, :-1, :]  # shift: predict next token
    labels = input_ids[1:]  # target tokens

    # Log softmax along vocab dimension
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Gather log probs for actual tokens
    token_logprobs = mx.take_along_axis(
        log_probs[0], labels[:, None], axis=-1
    ).squeeze(-1)
    return token_logprobs


def _compute_old_logprobs(examples: List[TrajectoryExample], model) -> None:
    """Compute reference log-probabilities with the current model (no grad)."""
    model.eval()
    for example in examples:
        input_ids = mx.array(example.input_ids)
        logprobs = _compute_logprobs(model, input_ids)
        mx.eval(logprobs)
        example.old_logprobs = logprobs.tolist()
    model.train()


# ---------------------------------------------------------------------------
# Assistant target mask
# ---------------------------------------------------------------------------

def _assistant_target_mask(example: TrajectoryExample) -> mx.array:
    """Boolean mask over shifted positions — True for assistant tokens."""
    length = max(0, len(example.input_ids) - 1)
    mask = mx.zeros((length,), dtype=mx.bool_)
    start = max(0, min(length, example.assistant_start - 1))
    if start < length:
        # Build mask: zeros before start, ones from start onwards
        ones = mx.ones((length - start,), dtype=mx.bool_)
        zeros = mx.zeros((start,), dtype=mx.bool_)
        mask = mx.concatenate([zeros, ones])
    return mask


# ---------------------------------------------------------------------------
# MIS-PO loss (single trajectory)
# ---------------------------------------------------------------------------

def _mis_po_loss(
    model,
    input_ids: mx.array,
    old_logprobs: mx.array,
    target_mask: mx.array,
    reward: float,
    token_ratio_min: float,
    token_ratio_max: float,
    traj_ratio_min: float,
    traj_ratio_max: float,
    kl_beta: float,
) -> mx.array:
    """Compute MIS-PO loss for a single trajectory."""
    logits = model(input_ids[None, :])
    logits = logits[:, :-1, :]
    labels = input_ids[1:]

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    new_logprobs = mx.take_along_axis(
        log_probs[0], labels[:, None], axis=-1
    ).squeeze(-1)

    log_ratio = new_logprobs - old_logprobs
    ratio = mx.exp(log_ratio)

    # Token-level filtering
    token_mask = target_mask & (ratio >= token_ratio_min) & (ratio <= token_ratio_max)
    token_mask_any = mx.any(token_mask)

    # Trajectory-level filtering
    masked_log_ratio = mx.where(target_mask, log_ratio, mx.zeros_like(log_ratio))
    target_count = mx.sum(target_mask.astype(mx.float32))
    trajectory_ratio = mx.exp(
        mx.sum(masked_log_ratio) / mx.maximum(target_count, mx.array(1.0))
    )
    traj_in_range = (trajectory_ratio >= traj_ratio_min) & (trajectory_ratio <= traj_ratio_max)

    # Actor loss: -mean(new_logprobs[token_mask]) * reward
    masked_logprobs = mx.where(token_mask, new_logprobs, mx.zeros_like(new_logprobs))
    token_count = mx.maximum(mx.sum(token_mask.astype(mx.float32)), mx.array(1.0))
    actor_loss = -(mx.sum(masked_logprobs) / token_count) * reward

    # KL loss: E[exp(log_ratio) - log_ratio - 1] over target tokens
    kl_terms = mx.exp(log_ratio) - log_ratio - 1.0
    masked_kl = mx.where(target_mask, kl_terms, mx.zeros_like(kl_terms))
    kl_loss = mx.sum(masked_kl) / mx.maximum(target_count, mx.array(1.0))

    # Zero out loss if masks reject this trajectory
    loss = (actor_loss + kl_beta * kl_loss)
    loss = mx.where(token_mask_any & traj_in_range, loss, mx.array(0.0))

    return loss


# ---------------------------------------------------------------------------
# Adapter save / next-dir helpers
# ---------------------------------------------------------------------------

def _next_adapter_dir(cfg: Dict[str, Any]) -> Path:
    root = Path(str(cfg.get("adapter_output_dir") or "")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith("adapter_")
    )
    next_idx = 1
    if existing:
        try:
            next_idx = max(int(path.name.split("_", 1)[1]) for path in existing) + 1
        except Exception:
            next_idx = len(existing) + 1
    return root / f"adapter_{next_idx:05d}"


def _cleanup_old_adapters(cfg: Dict[str, Any], active_adapter_path: str) -> None:
    keep = max(1, int(cfg.get("max_saved_adapters", 4)))
    root = Path(str(cfg.get("adapter_output_dir") or "")).expanduser()
    active_path = Path(active_adapter_path).expanduser()
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("adapter_")),
        key=lambda path: path.name,
    )
    stale = [path for path in candidates if path != active_path]
    while len(stale) > max(0, keep - 1):
        victim = stale.pop(0)
        shutil.rmtree(victim, ignore_errors=True)


def _save_mlx_adapter(model, adapter_dir: Path, lora_cfg: dict) -> None:
    """Save LoRA adapter weights in mlx-lm compatible format."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Collect only LoRA parameters
    lora_weights = {}
    for name, param in tree_flatten(model.trainable_parameters()):
        lora_weights[name] = param

    # Save as safetensors
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), lora_weights)

    # Save LoRA config for mlx-lm compatibility
    adapter_config = {
        "lora_layers": lora_cfg.get("num_layers", -1),
        "lora_parameters": lora_cfg.get("lora_parameters", {}),
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_batch(
    export_path: str | Path,
    *,
    runtime_base_url: Optional[str] = None,
    feedback_ids: Optional[List[int]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one online-RL batch with MIS-PO on MLX (Apple Silicon)."""
    if not MLX_AVAILABLE:
        raise RuntimeError(
            "MLX is not available. Install with: pip install 'hermes-agent[online-rl-mlx]'"
        )

    cfg = cfg or load_online_rl_config()
    export_file = Path(export_path).expanduser()
    rows = [
        json.loads(line)
        for line in export_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if feedback_ids:
        wanted = {int(fid) for fid in feedback_ids}
        rows = [row for row in rows if int(row["feedback_id"]) in wanted]
    if not rows:
        raise RuntimeError(f"No trainable online-RL rows found in {export_file}")

    model, tokenizer, base_model_name, lora_cfg = _load_training_model(cfg)
    examples = _build_examples(rows, tokenizer, cfg)
    if not examples:
        raise RuntimeError(
            "No usable assistant trajectories were found in the exported feedback batch"
        )

    # Compute reference log-probabilities
    _compute_old_logprobs(examples, model)

    # Optimizer
    lr = float(cfg.get("learning_rate", 2e-6))
    weight_decay = float(cfg.get("weight_decay", 0.1))
    warmup_steps = max(0, int(cfg.get("warmup_steps", 20)))
    total_steps = max(1, int(cfg.get("train_steps", 16)))
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 4)))

    # Linear warmup schedule
    if warmup_steps > 0:
        scheduler = optim.linear_schedule(
            init=1e-7, end=lr, steps=warmup_steps
        )
        optimizer = optim.AdamW(
            learning_rate=scheduler,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

    # MIS-PO hyperparams
    token_ratio_min = float(cfg.get("token_ratio_min", 0.5))
    token_ratio_max = float(cfg.get("token_ratio_max", 2.0))
    traj_ratio_min = float(cfg.get("trajectory_ratio_min", 0.996))
    traj_ratio_max = float(cfg.get("trajectory_ratio_max", 1.001))
    kl_beta = float(cfg.get("kl_coefficient", 0.001))

    stats = {
        "steps": total_steps,
        "used_examples": len(examples),
        "masked_updates": 0,
        "applied_updates": 0,
        "mean_reward": sum(e.reward for e in examples) / float(len(examples)),
        "backend": "mlx",
    }

    # Build the loss function for value_and_grad
    def step_loss(model, input_ids, old_lp, tmask, reward):
        return _mis_po_loss(
            model, input_ids, old_lp, tmask, reward,
            token_ratio_min, token_ratio_max,
            traj_ratio_min, traj_ratio_max,
            kl_beta,
        )

    loss_and_grad_fn = nn.value_and_grad(model, step_loss)

    accumulated_grads = None
    accum_count = 0

    for step in range(total_steps):
        example = examples[step % len(examples)]
        input_ids = mx.array(example.input_ids)
        old_logprobs = mx.array(example.old_logprobs or [])
        target_mask = _assistant_target_mask(example)

        if not mx.any(target_mask).item():
            stats["masked_updates"] += 1
            continue

        loss, grads = loss_and_grad_fn(
            model, input_ids, old_logprobs, target_mask, example.reward
        )
        mx.eval(loss)

        loss_val = loss.item()
        if loss_val == 0.0:
            stats["masked_updates"] += 1
            continue

        # Scale gradients for accumulation
        if grad_accum > 1:
            grads = tree_unflatten(
                [(k, v / grad_accum) for k, v in tree_flatten(grads)]
            )

        # Accumulate
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = tree_unflatten(
                [(k, a + b) for (k, a), (_, b) in
                 zip(tree_flatten(accumulated_grads), tree_flatten(grads))]
            )
        accum_count += 1

        # Apply update
        if accum_count >= grad_accum or (step + 1) == total_steps:
            if accumulated_grads is not None:
                # Gradient clipping (max norm 1.0)
                grad_list = tree_flatten(accumulated_grads)
                total_norm_sq = sum(
                    mx.sum(mx.square(v)).item() for _, v in grad_list
                )
                total_norm = math.sqrt(total_norm_sq)
                if total_norm > 1.0:
                    scale = 1.0 / total_norm
                    accumulated_grads = tree_unflatten(
                        [(k, v * scale) for k, v in grad_list]
                    )

                optimizer.update(model, accumulated_grads)
                mx.eval(model.parameters(), optimizer.state)
                stats["applied_updates"] += 1
            accumulated_grads = None
            accum_count = 0

    # Save adapter
    adapter_dir = _next_adapter_dir(cfg)
    _save_mlx_adapter(model, adapter_dir, lora_cfg)

    # Also save tokenizer config for completeness
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(adapter_dir))

    # Publish adapter
    published_state = publish_online_rl_adapter(
        str(adapter_dir),
        runtime_base_url=runtime_base_url,
        base_model=base_model_name,
        cfg=cfg,
    )
    _cleanup_old_adapters(cfg, str(adapter_dir))

    stats.update(
        {
            "adapter_dir": str(adapter_dir),
            "published_model": published_state.get("active_model_name"),
            "published_backend": published_state.get("backend"),
            "algorithm": str(cfg.get("algorithm") or "mis_po"),
        }
    )
    return stats
