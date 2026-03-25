"""Tinker API-based trainer for online LoRA updates on remote SOTA models.

Delegates all compute (forward/backward, optimizer steps, checkpointing) to the
Tinker API while reusing the same MIS-PO-style importance-sampled policy gradient
algorithm used by the local PyTorch and MLX trainers.

Supported models include moonshotai/Kimi-K2.5 and
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Tinker model IDs that are known to be available
TINKER_SUPPORTED_MODELS = {
    "moonshotai/Kimi-K2.5",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}


def _normalize_messages(messages):
    """Normalize message content to plain strings (mirrors online_rl_trainer.py)."""
    normalized = []
    for message in messages or []:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in {"text", "input_text", "output_text"}:
                        parts.append(str(item.get("text") or ""))
                elif item is not None:
                    parts.append(str(item))
            content = "\n".join(part for part in parts if part)
        normalized.append({"role": role, "content": str(content)})
    return normalized


def _chat_text(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    """Apply chat template or fallback to plain formatting."""
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    except Exception:
        pass
    lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        lines.append(f"{role}: {msg.get('content', '')}")
    if add_generation_prompt:
        lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _build_tinker_examples(
    export_rows: List[Dict[str, Any]],
    tokenizer,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert feedback JSONL rows to tokenized examples for Tinker training.

    Returns a list of dicts with keys:
        feedback_id, reward, input_ids, assistant_start, prompt_ids
    """
    max_length = max(512, int(cfg.get("max_sequence_length", 4096)))
    examples = []
    for row in export_rows:
        reward = float(row.get("reward") or 0.0)
        if reward == 0:
            continue

        prompt_messages = _normalize_messages(row.get("prompt_messages") or [])
        assistant_message = row.get("assistant_message") or {}
        assistant_content = str(
            assistant_message.get("content") or row.get("response_text") or ""
        ).strip()
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

        examples.append({
            "feedback_id": int(row["feedback_id"]),
            "reward": reward,
            "input_ids": list(full_ids),
            "assistant_start": int(assistant_start),
        })
    return examples


def _compute_ref_logprobs(sampling_client, examples: List[Dict[str, Any]]) -> None:
    """Compute reference (old policy) logprobs for each example via Tinker sampling client."""
    import tinker
    from tinker import types

    for example in examples:
        prompt = tinker.ModelInput.from_ints(example["input_ids"])
        logprobs = sampling_client.compute_logprobs(prompt).result()
        # logprobs is a list of float|None per token; first token has no logprob
        example["old_logprobs"] = [
            lp if lp is not None else 0.0 for lp in logprobs
        ]


def _build_datum(example: Dict[str, Any], *, loss_fn: str) -> "tinker.types.Datum":
    """Build a Tinker Datum from a tokenized example for the given loss function."""
    import tinker
    from tinker import types

    input_ids = example["input_ids"]
    assistant_start = example["assistant_start"]
    reward = example["reward"]
    old_logprobs = example.get("old_logprobs") or []
    n_tokens = len(input_ids)

    # Shifted targets (next-token prediction)
    target_tokens = np.array(input_ids[1:], dtype=np.int64)

    # Weight mask: only train on assistant tokens
    weights = np.zeros(n_tokens - 1, dtype=np.float32)
    start_idx = max(0, assistant_start - 1)
    weights[start_idx:] = 1.0

    loss_fn_inputs = {
        "target_tokens": types.TensorData(data=target_tokens.tolist(), shape=list(target_tokens.shape)),
        "weights": types.TensorData(data=weights.tolist(), shape=list(weights.shape)),
    }

    if loss_fn == "importance_sampling":
        # Importance sampling needs old logprobs and advantages
        # old_logprobs correspond to token positions; shift to align with targets
        ref_lps = old_logprobs[1:] if len(old_logprobs) > 1 else [0.0] * (n_tokens - 1)
        ref_lps_arr = np.array(ref_lps[: n_tokens - 1], dtype=np.float32)
        if len(ref_lps_arr) < n_tokens - 1:
            ref_lps_arr = np.pad(ref_lps_arr, (0, n_tokens - 1 - len(ref_lps_arr)))

        # Advantages: reward applied to assistant tokens only
        advantages = np.zeros(n_tokens - 1, dtype=np.float32)
        advantages[start_idx:] = reward
        loss_fn_inputs["logprobs"] = types.TensorData(
            data=ref_lps_arr.tolist(), shape=list(ref_lps_arr.shape)
        )
        loss_fn_inputs["advantages"] = types.TensorData(
            data=advantages.tolist(), shape=list(advantages.shape)
        )

    model_input = tinker.ModelInput.from_ints(input_ids)
    datum = types.Datum(
        model_input=model_input,
        loss_fn_inputs=loss_fn_inputs,
    )
    datum.convert_tensors()
    return datum


def train_batch(
    export_path: str | Path,
    *,
    runtime_base_url: Optional[str] = None,
    feedback_ids: Optional[List[int]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one online-RL batch via the Tinker API.

    Same interface as the PyTorch and MLX trainers, but all compute runs
    remotely through Tinker's forward_backward / optim_step API.
    """
    import tinker
    from tinker import types

    from agent.online_rl import (
        load_online_rl_config,
        load_online_rl_state,
        publish_online_rl_adapter,
    )

    cfg = cfg or load_online_rl_config()

    # Parse feedback export
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

    # Resolve Tinker model and config
    tinker_model = str(cfg.get("tinker_base_model") or cfg.get("training_base_model") or "").strip()
    if not tinker_model:
        raise RuntimeError(
            "tinker_base_model (or training_base_model) is required for Tinker training"
        )
    lora_rank = int(cfg.get("tinker_lora_rank") or cfg.get("lora_rank", 32))
    loss_fn = str(cfg.get("tinker_loss_fn") or "importance_sampling").strip()
    learning_rate = float(cfg.get("learning_rate", 2e-6))
    weight_decay = float(cfg.get("weight_decay", 0.1))
    total_steps = max(1, int(cfg.get("train_steps", 16)))
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 4)))

    api_key = str(cfg.get("tinker_api_key") or "").strip()
    if not api_key:
        raise RuntimeError(
            "TINKER_API_KEY environment variable or tinker_api_key config is required"
        )

    # Create Tinker clients
    logger.info("Connecting to Tinker API for model %s (rank=%d, loss=%s)", tinker_model, lora_rank, loss_fn)
    service_client = tinker.ServiceClient()

    # Check if we should resume from an existing checkpoint
    runtime_state = load_online_rl_state(cfg)
    existing_ckpt = str(runtime_state.get("tinker_checkpoint_path") or cfg.get("tinker_checkpoint_path") or "").strip()

    if existing_ckpt and existing_ckpt.startswith("tinker://"):
        logger.info("Resuming from checkpoint: %s", existing_ckpt)
        try:
            training_client = service_client.create_training_client_from_state(existing_ckpt)
        except Exception as exc:
            logger.warning("Failed to resume from checkpoint %s: %s. Starting fresh.", existing_ckpt, exc)
            training_client = service_client.create_lora_training_client(
                base_model=tinker_model,
                rank=lora_rank,
            )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=tinker_model,
            rank=lora_rank,
        )

    tokenizer = training_client.get_tokenizer()

    # Build examples
    examples = _build_tinker_examples(rows, tokenizer, cfg)
    if not examples:
        raise RuntimeError("No usable assistant trajectories found in the exported feedback batch")

    # Compute reference logprobs for importance sampling
    if loss_fn in ("importance_sampling", "cispo", "ppo"):
        logger.info("Computing reference logprobs for %d examples...", len(examples))
        sampling_client = training_client.save_weights_and_get_sampling_client(name="ref_policy")
        _compute_ref_logprobs(sampling_client, examples)

    # Training loop
    logger.info("Starting training: %d steps, %d examples, grad_accum=%d", total_steps, len(examples), grad_accum)
    t0 = time.time()
    stats = {
        "steps": total_steps,
        "used_examples": len(examples),
        "masked_updates": 0,
        "applied_updates": 0,
        "mean_reward": sum(ex["reward"] for ex in examples) / float(len(examples)),
    }

    loss_fn_config = {}
    if loss_fn in ("ppo", "cispo"):
        loss_fn_config = {"clip_low_threshold": 0.2, "clip_high_threshold": 0.2}

    for step in range(total_steps):
        example = examples[step % len(examples)]
        datum = _build_datum(example, loss_fn=loss_fn)

        fwdbwd_future = training_client.forward_backward([datum], loss_fn, loss_fn_config or None)
        fwdbwd_result = fwdbwd_future.result()

        # Accumulate gradients; step optimizer every grad_accum steps
        if (step + 1) % grad_accum == 0 or (step + 1) == total_steps:
            optim_future = training_client.optim_step(
                types.AdamParams(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    grad_clip_norm=1.0,
                )
            )
            optim_future.result()
            stats["applied_updates"] += 1

        if (step + 1) % max(1, total_steps // 4) == 0:
            logger.info("Step %d/%d completed", step + 1, total_steps)

    elapsed = time.time() - t0
    logger.info("Training completed in %.1fs", elapsed)

    # Save checkpoint for inference
    adapter_name = str(cfg.get("adapter_name") or "hermes-online-rl").strip()
    ckpt_name = f"{adapter_name}_{int(time.time())}"
    save_result = training_client.save_weights_for_sampler(name=ckpt_name).result()
    checkpoint_path = save_result.path
    logger.info("Saved Tinker checkpoint: %s", checkpoint_path)

    # Publish adapter (stores state)
    published_state = publish_online_rl_adapter(
        checkpoint_path,
        runtime_base_url=runtime_base_url,
        base_model=tinker_model,
        cfg=cfg,
    )

    stats.update({
        "adapter_dir": checkpoint_path,
        "tinker_checkpoint_path": checkpoint_path,
        "published_model": published_state.get("active_model_name"),
        "backend": "tinker",
        "algorithm": str(cfg.get("algorithm") or "mis_po"),
        "tinker_base_model": tinker_model,
        "elapsed_seconds": elapsed,
    })
    return stats
