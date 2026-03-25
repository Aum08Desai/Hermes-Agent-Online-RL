"""Built-in MIS-PO trainer for online LoRA updates from Hermes feedback batches."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.online_rl import (
    load_online_rl_config,
    load_online_rl_state,
    publish_online_rl_adapter,
)
from hermes_state import SessionDB


def _should_use_tinker(cfg: Dict[str, Any]) -> bool:
    """Return True when training should use the Tinker API backend."""
    backend = str(cfg.get("backend") or "auto").strip().lower()
    if backend == "tinker":
        return True
    # Auto-detect: if tinker_api_key and tinker_base_model are configured
    if backend in {"", "auto"} and cfg.get("tinker_api_key") and cfg.get("tinker_base_model"):
        return True
    return False


def _should_use_mlx(cfg: Dict[str, Any]) -> bool:
    """Return True when training should use the MLX backend."""
    device = str(cfg.get("device") or "auto").strip().lower()
    # Explicit MLX request
    if device == "mlx":
        return True
    # Auto-detect: prefer MLX on Apple Silicon when available
    if device in {"", "auto"} and platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            from agent.online_rl_trainer_mlx import is_mlx_available
            return is_mlx_available()
        except ImportError:
            return False
    return False


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


def _device_for_training(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device") or "auto").strip().lower()
    if requested not in {"", "auto"}:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_dtype_for_training(cfg: Dict[str, Any], device: torch.device):
    requested = str(cfg.get("torch_dtype") or "auto").strip().lower()
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16", "half"}:
        return torch.float16
    if requested in {"fp32", "float32"}:
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
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


def _resolve_target_modules(model, requested: List[str]) -> List[str]:
    available = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    selected = [name for name in requested if name in available]
    if selected:
        return selected
    fallbacks = [name for name in available if name.endswith("_proj") or name in {"gate_proj", "up_proj", "down_proj"}]
    if fallbacks:
        return sorted(fallbacks)
    return sorted(name for name in available if name != "lm_head")


def _load_training_model(cfg: Dict[str, Any], *, device: torch.device):
    runtime_state = load_online_rl_state(cfg)
    base_model_name = str(runtime_state.get("base_model_name") or cfg.get("training_base_model") or "").strip()
    if not base_model_name:
        raise RuntimeError(
            "online_rl.training_base_model is required for built-in LoRA training"
        )

    dtype = _torch_dtype_for_training(cfg, device)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )

    active_adapter_path = str(runtime_state.get("active_adapter_path") or "").strip()
    if active_adapter_path and Path(active_adapter_path).exists():
        model = PeftModel.from_pretrained(model, active_adapter_path, is_trainable=True)
    else:
        target_modules = _resolve_target_modules(model, list(cfg.get("target_modules") or []))
        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_rank", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

    model.to(device)
    model.train()
    return model, tokenizer, base_model_name


def _example_logprobs(model, example: TrajectoryExample, device: torch.device) -> torch.Tensor:
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
    return token_logprobs.detach().cpu()


def _assistant_target_mask(example: TrajectoryExample, device: torch.device) -> torch.Tensor:
    length = max(0, len(example.input_ids) - 1)
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    start = max(0, min(length, example.assistant_start - 1))
    mask[start:] = True
    return mask


def _compute_old_logprobs(examples: List[TrajectoryExample], model, device: torch.device) -> None:
    model.eval()
    for example in examples:
        example.old_logprobs = _example_logprobs(model, example, device).tolist()
    model.train()


def _next_adapter_dir(cfg: Dict[str, Any]) -> Path:
    root = Path(str(cfg.get("adapter_output_dir") or "")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("adapter_"))
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
    candidates = sorted((path for path in root.iterdir() if path.is_dir() and path.name.startswith("adapter_")), key=lambda path: path.name)
    stale = [path for path in candidates if path != active_path]
    while len(stale) > max(0, keep - 1):
        victim = stale.pop(0)
        shutil.rmtree(victim, ignore_errors=True)


def _optimizer_and_scheduler(model, cfg: Dict[str, Any]):
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg.get("learning_rate", 2e-6)),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
    )
    warmup_steps = max(0, int(cfg.get("warmup_steps", 20)))

    def _lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    return optimizer, scheduler


def train_batch(
    export_path: str | Path,
    *,
    runtime_base_url: Optional[str] = None,
    feedback_ids: Optional[List[int]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one online-RL batch with a single-trajectory MIS-PO objective."""
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

    device = _device_for_training(cfg)
    model, tokenizer, base_model_name = _load_training_model(cfg, device=device)
    examples = _build_examples(rows, tokenizer, cfg)
    if not examples:
        raise RuntimeError("No usable assistant trajectories were found in the exported feedback batch")

    _compute_old_logprobs(examples, model, device)
    optimizer, scheduler = _optimizer_and_scheduler(model, cfg)

    token_ratio_min = float(cfg.get("token_ratio_min", 0.5))
    token_ratio_max = float(cfg.get("token_ratio_max", 2.0))
    traj_ratio_min = float(cfg.get("trajectory_ratio_min", 0.996))
    traj_ratio_max = float(cfg.get("trajectory_ratio_max", 1.001))
    kl_beta = float(cfg.get("kl_coefficient", 0.001))
    total_steps = max(1, int(cfg.get("train_steps", 16)))
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 4)))

    stats = {
        "steps": total_steps,
        "used_examples": len(examples),
        "masked_updates": 0,
        "applied_updates": 0,
        "mean_reward": sum(example.reward for example in examples) / float(len(examples)),
    }

    optimizer.zero_grad(set_to_none=True)
    for step in range(total_steps):
        example = examples[step % len(examples)]
        input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device).unsqueeze(0)
        old_logprobs = torch.tensor(example.old_logprobs or [], dtype=torch.float32, device=device)
        target_mask = _assistant_target_mask(example, device)
        if not torch.any(target_mask):
            continue

        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        new_logprobs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)

        log_ratio = (new_logprobs - old_logprobs).float()
        ratio = torch.exp(log_ratio)
        token_mask = target_mask & (ratio >= token_ratio_min) & (ratio <= token_ratio_max)
        if torch.any(target_mask):
            trajectory_ratio = torch.exp(log_ratio[target_mask].mean())
        else:
            trajectory_ratio = torch.tensor(1.0, device=device)
        trajectory_mask = 1.0 if traj_ratio_min <= float(trajectory_ratio.item()) <= traj_ratio_max else 0.0

        if trajectory_mask == 0.0 or not torch.any(token_mask):
            stats["masked_updates"] += 1
            continue

        advantage = float(example.reward)
        actor_loss = -(
            new_logprobs[token_mask].mean() * advantage * trajectory_mask
        )
        kl_estimate = torch.exp(log_ratio[target_mask]) - log_ratio[target_mask] - 1.0
        kl_loss = kl_estimate.mean() if kl_estimate.numel() else torch.tensor(0.0, device=device)
        loss = (actor_loss + kl_beta * kl_loss) / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == total_steps:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            stats["applied_updates"] += 1

    adapter_dir = _next_adapter_dir(cfg)
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

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
            "backend": published_state.get("backend"),
            "algorithm": str(cfg.get("algorithm") or "mis_po"),
        }
    )
    return stats


def run_train_batch(
    export_path: str | Path,
    *,
    runtime_base_url: Optional[str] = None,
    feedback_ids: Optional[List[int]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one batch and update RL feedback statuses in SQLite.

    Automatically routes to the MLX-native trainer on Apple Silicon when MLX
    is available (or when ``device`` is set to ``"mlx"`` in the config).
    """
    cfg = cfg or load_online_rl_config()
    ids = list(feedback_ids or [])
    if not ids:
        env_ids = os.getenv("HERMES_RL_FEEDBACK_IDS", "").strip()
        if env_ids:
            ids = [int(part) for part in env_ids.split(",") if part.strip()]

    # Select the appropriate train_batch implementation
    if _should_use_tinker(cfg):
        from agent.online_rl_trainer_tinker import train_batch as tinker_train_batch
        _train_fn = tinker_train_batch
    elif _should_use_mlx(cfg):
        from agent.online_rl_trainer_mlx import train_batch as mlx_train_batch
        _train_fn = mlx_train_batch
    else:
        _train_fn = train_batch

    db = SessionDB()
    try:
        if ids:
            db.mark_rl_feedback_status(ids, trainer_status="training", export_path=str(export_path), last_error=None)
        result = _train_fn(
            export_path,
            runtime_base_url=runtime_base_url,
            feedback_ids=ids or None,
            cfg=cfg,
        )
        if ids:
            db.mark_rl_feedback_status(
                ids,
                trainer_status="applied",
                export_path=result.get("adapter_dir"),
                last_error=None,
            )
        return result
    except Exception as exc:
        if ids:
            db.mark_rl_feedback_status(
                ids,
                trainer_status="failed",
                export_path=str(export_path),
                last_error=str(exc),
            )
        raise
    finally:
        db.close()

