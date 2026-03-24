"""Helpers for local online-RL feedback capture, MIS-PO training, and adapter activation."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import httpx

from agent.model_metadata import detect_local_server_type
from hermes_state import (
    RL_FEEDBACK_DOWNWEIGHT,
    RL_FEEDBACK_NO_RL,
    RL_FEEDBACK_UPWEIGHT,
    is_trainable_rl_feedback_label,
    normalize_rl_feedback_label,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_server_root(base_url: Optional[str]) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return parsed._replace(path=path, params="", query="", fragment="").geturl().rstrip("/")


def is_local_base_url(base_url: Optional[str]) -> bool:
    """Return True when a runtime URL points at a locally hosted inference server."""
    if not base_url:
        return False
    normalized = str(base_url).strip()
    parsed = urlparse(normalized if "://" in normalized else f"http://{normalized}")
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def load_online_rl_config() -> Dict[str, Any]:
    """Load merged online-RL config from config.yaml plus env overrides."""
    from hermes_cli.config import get_hermes_home, load_config

    root_cfg = load_config()
    cfg = root_cfg.get("online_rl") or {}
    model_cfg = root_cfg.get("model") if isinstance(root_cfg.get("model"), dict) else {}

    home = get_hermes_home()
    online_rl_home = home / "online_rl"

    result = {
        "enabled": bool(cfg.get("enabled", False)),
        "prompt_after_response": bool(cfg.get("prompt_after_response", False)),
        "local_only": bool(cfg.get("local_only", True)),
        "backend": str(cfg.get("backend") or "auto").strip().lower(),
        "algorithm": str(cfg.get("algorithm") or "mis_po").strip().lower(),
        "min_batch_size": _safe_int(cfg.get("min_batch_size", 8), 8),
        "max_batch_size": _safe_int(cfg.get("max_batch_size", 64), 64),
        "feedback_timeout_seconds": _safe_int(cfg.get("feedback_timeout_seconds", 45), 45),
        "export_dir": str(Path(cfg.get("export_dir") or (online_rl_home / "exports")).expanduser()),
        "adapter_output_dir": str(Path(cfg.get("adapter_output_dir") or (online_rl_home / "adapters")).expanduser()),
        "state_path": str(Path(cfg.get("state_path") or (online_rl_home / "state.json")).expanduser()),
        "trainer_command": str(cfg.get("trainer_command") or "").strip(),
        "builtin_trainer": bool(cfg.get("builtin_trainer", True)),
        "model_base_url": str((model_cfg or {}).get("base_url") or "").strip(),
        "training_base_model": str(cfg.get("training_base_model") or (model_cfg or {}).get("default") or "").strip(),
        "adapter_name": str(cfg.get("adapter_name") or "hermes-online-rl").strip(),
        "ollama_model_name": str(cfg.get("ollama_model_name") or "").strip(),
        "ollama_command": str(cfg.get("ollama_command") or "ollama").strip(),
        "train_steps": _safe_int(cfg.get("train_steps", 16), 16),
        "micro_batch_size": _safe_int(cfg.get("micro_batch_size", 1), 1),
        "gradient_accumulation_steps": _safe_int(cfg.get("gradient_accumulation_steps", 4), 4),
        "learning_rate": _safe_float(cfg.get("learning_rate", 2e-6), 2e-6),
        "weight_decay": _safe_float(cfg.get("weight_decay", 0.1), 0.1),
        "warmup_steps": _safe_int(cfg.get("warmup_steps", 20), 20),
        "kl_coefficient": _safe_float(cfg.get("kl_coefficient", 0.001), 0.001),
        "token_ratio_min": _safe_float(cfg.get("token_ratio_min", 0.8), 0.8),
        "token_ratio_max": _safe_float(cfg.get("token_ratio_max", 1.25), 1.25),
        "trajectory_ratio_min": _safe_float(cfg.get("trajectory_ratio_min", 0.9), 0.9),
        "trajectory_ratio_max": _safe_float(cfg.get("trajectory_ratio_max", 1.1), 1.1),
        "max_sequence_length": _safe_int(cfg.get("max_sequence_length", 4096), 4096),
        "lora_rank": _safe_int(cfg.get("lora_rank", 16), 16),
        "lora_alpha": _safe_int(cfg.get("lora_alpha", 32), 32),
        "lora_dropout": _safe_float(cfg.get("lora_dropout", 0.05), 0.05),
        "target_modules": list(
            cfg.get("target_modules")
            or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        "device": str(cfg.get("device") or "auto").strip(),
        "torch_dtype": str(cfg.get("torch_dtype") or "auto").strip(),
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
        "max_saved_adapters": _safe_int(cfg.get("max_saved_adapters", 4), 4),
    }

    result["enabled"] = _env_bool("HERMES_ONLINE_RL_ENABLED", result["enabled"])
    result["prompt_after_response"] = _env_bool(
        "HERMES_ONLINE_RL_PROMPT_AFTER_RESPONSE",
        result["prompt_after_response"],
    )
    result["local_only"] = _env_bool("HERMES_ONLINE_RL_LOCAL_ONLY", result["local_only"])
    result["builtin_trainer"] = _env_bool("HERMES_ONLINE_RL_BUILTIN_TRAINER", result["builtin_trainer"])
    result["backend"] = os.getenv("HERMES_ONLINE_RL_BACKEND", result["backend"]).strip().lower() or "auto"
    result["algorithm"] = os.getenv("HERMES_ONLINE_RL_ALGORITHM", result["algorithm"]).strip().lower() or "mis_po"
    result["min_batch_size"] = _safe_int(os.getenv("HERMES_ONLINE_RL_MIN_BATCH_SIZE"), result["min_batch_size"])
    result["max_batch_size"] = _safe_int(os.getenv("HERMES_ONLINE_RL_MAX_BATCH_SIZE"), result["max_batch_size"])
    result["feedback_timeout_seconds"] = _safe_int(
        os.getenv("HERMES_ONLINE_RL_FEEDBACK_TIMEOUT_SECONDS"),
        result["feedback_timeout_seconds"],
    )
    result["export_dir"] = os.getenv("HERMES_ONLINE_RL_EXPORT_DIR", result["export_dir"])
    result["adapter_output_dir"] = os.getenv("HERMES_ONLINE_RL_ADAPTER_OUTPUT_DIR", result["adapter_output_dir"])
    result["state_path"] = os.getenv("HERMES_ONLINE_RL_STATE_PATH", result["state_path"])
    result["trainer_command"] = os.getenv("HERMES_ONLINE_RL_TRAINER_COMMAND", result["trainer_command"]).strip()
    result["training_base_model"] = os.getenv(
        "HERMES_ONLINE_RL_TRAINING_BASE_MODEL",
        result["training_base_model"],
    ).strip()
    result["adapter_name"] = os.getenv("HERMES_ONLINE_RL_ADAPTER_NAME", result["adapter_name"]).strip() or "hermes-online-rl"
    result["ollama_model_name"] = os.getenv("HERMES_ONLINE_RL_OLLAMA_MODEL_NAME", result["ollama_model_name"]).strip()
    result["train_steps"] = _safe_int(os.getenv("HERMES_ONLINE_RL_TRAIN_STEPS"), result["train_steps"])
    result["learning_rate"] = _safe_float(os.getenv("HERMES_ONLINE_RL_LEARNING_RATE"), result["learning_rate"])
    result["weight_decay"] = _safe_float(os.getenv("HERMES_ONLINE_RL_WEIGHT_DECAY"), result["weight_decay"])
    result["warmup_steps"] = _safe_int(os.getenv("HERMES_ONLINE_RL_WARMUP_STEPS"), result["warmup_steps"])
    result["kl_coefficient"] = _safe_float(os.getenv("HERMES_ONLINE_RL_KL_COEFFICIENT"), result["kl_coefficient"])
    result["token_ratio_min"] = _safe_float(os.getenv("HERMES_ONLINE_RL_TOKEN_RATIO_MIN"), result["token_ratio_min"])
    result["token_ratio_max"] = _safe_float(os.getenv("HERMES_ONLINE_RL_TOKEN_RATIO_MAX"), result["token_ratio_max"])
    result["trajectory_ratio_min"] = _safe_float(
        os.getenv("HERMES_ONLINE_RL_TRAJECTORY_RATIO_MIN"),
        result["trajectory_ratio_min"],
    )
    result["trajectory_ratio_max"] = _safe_float(
        os.getenv("HERMES_ONLINE_RL_TRAJECTORY_RATIO_MAX"),
        result["trajectory_ratio_max"],
    )
    result["max_sequence_length"] = _safe_int(
        os.getenv("HERMES_ONLINE_RL_MAX_SEQUENCE_LENGTH"),
        result["max_sequence_length"],
    )
    return result


def online_rl_enabled_for_runtime(
    *,
    runtime_base_url: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when online RL is enabled for the resolved runtime."""
    cfg = cfg or load_online_rl_config()
    if not cfg.get("enabled"):
        return False
    effective_base_url = runtime_base_url or cfg.get("model_base_url")
    if cfg.get("local_only", True) and not is_local_base_url(effective_base_url):
        return False
    return True


def prompt_for_feedback_enabled(
    *,
    runtime_base_url: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when the interactive post-response feedback prompt should appear."""
    cfg = cfg or load_online_rl_config()
    return bool(cfg.get("prompt_after_response")) and online_rl_enabled_for_runtime(
        runtime_base_url=runtime_base_url,
        cfg=cfg,
    )


def resolve_online_rl_backend(
    *,
    runtime_base_url: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve the local serving backend used for adapter activation."""
    cfg = cfg or load_online_rl_config()
    backend = str(cfg.get("backend") or "auto").strip().lower()
    if backend and backend != "auto":
        return backend
    base_url = runtime_base_url or cfg.get("model_base_url")
    if not base_url:
        return None
    try:
        detected = detect_local_server_type(base_url)
    except Exception:
        detected = None
    if detected in {"vllm", "ollama", "mlx"}:
        return detected
    return None


def load_online_rl_state(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load persisted online-RL runtime state from disk."""
    cfg = cfg or load_online_rl_config()
    path = Path(str(cfg.get("state_path") or "")).expanduser()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_online_rl_state(state: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Atomically persist online-RL runtime state."""
    cfg = cfg or load_online_rl_config()
    path = Path(str(cfg.get("state_path") or "")).expanduser()
    _ensure_parent_dir(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
    return state


def _post_json(url: str, payload: Dict[str, Any]) -> None:
    with httpx.Client(timeout=10.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()


def _load_vllm_adapter(
    *,
    runtime_base_url: str,
    adapter_name: str,
    adapter_path: str,
) -> None:
    server_root = _normalize_server_root(runtime_base_url)
    if not server_root:
        raise ValueError("runtime_base_url is required for vLLM adapter activation")

    payloads = [
        {"lora_name": adapter_name, "lora_path": adapter_path, "load_inplace": True},
        {"lora_name": adapter_name, "lora_path": adapter_path},
    ]
    last_error: Optional[Exception] = None
    for payload in payloads:
        try:
            _post_json(f"{server_root}/v1/load_lora_adapter", payload)
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load vLLM LoRA adapter '{adapter_name}': {last_error}")


def _publish_ollama_adapter_model(
    *,
    model_name: str,
    base_model: str,
    adapter_path: str,
    cfg: Dict[str, Any],
) -> None:
    ollama_command = str(cfg.get("ollama_command") or "ollama").strip() or "ollama"
    if not base_model:
        raise ValueError("training_base_model is required to publish an Ollama adapter model")

    with tempfile.TemporaryDirectory(prefix="hermes_online_rl_ollama_") as tmpdir:
        modelfile_path = Path(tmpdir) / "Modelfile"
        modelfile_path.write_text(
            f"FROM {base_model}\nADAPTER {Path(adapter_path).expanduser()}\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [ollama_command, "create", model_name, "-f", str(modelfile_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"ollama create failed: {stderr}")


def publish_online_rl_adapter(
    adapter_path: str,
    *,
    runtime_base_url: Optional[str] = None,
    base_model: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Publish a freshly trained adapter to the active local runtime."""
    cfg = cfg or load_online_rl_config()
    backend = resolve_online_rl_backend(runtime_base_url=runtime_base_url, cfg=cfg)
    if backend not in {"vllm", "ollama", "mlx", None}:
        raise RuntimeError(f"Online RL backend '{backend}' does not support adapter publication")

    active_model_name = str(cfg.get("adapter_name") or "hermes-online-rl").strip()
    effective_base_model = str(base_model or cfg.get("training_base_model") or "").strip()
    if backend == "vllm":
        _load_vllm_adapter(
            runtime_base_url=runtime_base_url or str(cfg.get("model_base_url") or ""),
            adapter_name=active_model_name,
            adapter_path=adapter_path,
        )
    elif backend == "ollama":
        active_model_name = str(
            cfg.get("ollama_model_name") or f"{cfg.get('adapter_name') or 'hermes-online-rl'}:latest"
        ).strip()
        _publish_ollama_adapter_model(
            model_name=active_model_name,
            base_model=effective_base_model,
            adapter_path=adapter_path,
            cfg=cfg,
        )
    elif backend == "mlx":
        # MLX adapters are loaded directly from disk via mlx-lm's load()
        # with adapter_path=. No runtime server push needed.
        pass

    state = load_online_rl_state(cfg)
    state.update(
        {
            "algorithm": str(cfg.get("algorithm") or "mis_po"),
            "backend": backend or "mlx",
            "active_model_name": active_model_name,
            "active_adapter_path": str(Path(adapter_path).expanduser()),
            "base_model_name": effective_base_model,
            "updated_at": time.time(),
        }
    )
    save_online_rl_state(state, cfg)
    return state


def resolve_online_rl_model(
    base_model: str,
    *,
    runtime_base_url: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the effective inference model when an online-RL adapter is active."""
    cfg = cfg or load_online_rl_config()
    resolved = {
        "model": base_model,
        "active": False,
        "backend": resolve_online_rl_backend(runtime_base_url=runtime_base_url, cfg=cfg),
        "state": load_online_rl_state(cfg),
    }
    if not online_rl_enabled_for_runtime(runtime_base_url=runtime_base_url, cfg=cfg):
        return resolved

    state = resolved["state"] or {}
    active_model_name = str(state.get("active_model_name") or "").strip()
    active_adapter_path = str(state.get("active_adapter_path") or "").strip()
    backend = resolved["backend"]
    if not active_model_name:
        return resolved

    if backend == "vllm" and active_adapter_path:
        _load_vllm_adapter(
            runtime_base_url=runtime_base_url or str(cfg.get("model_base_url") or ""),
            adapter_name=active_model_name,
            adapter_path=active_adapter_path,
        )
        resolved["model"] = active_model_name
        resolved["active"] = True
        return resolved

    if backend == "ollama":
        resolved["model"] = active_model_name
        resolved["active"] = True
        return resolved

    if backend == "mlx" and active_adapter_path:
        # MLX adapters are loaded from disk; mark as active so callers know
        resolved["model"] = active_model_name
        resolved["active"] = True
        resolved["mlx_adapter_path"] = active_adapter_path
        return resolved

    return resolved


def _builtin_trainer_invocation(
    export_path: Path,
    *,
    runtime_base_url: Optional[str] = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "online-rl",
        "train-batch",
        "--export-path",
        str(export_path),
    ]
    if runtime_base_url:
        cmd.extend(["--runtime-base-url", runtime_base_url])
    return cmd


def _configured_trainer_invocation(trainer_command: str) -> str | list[str]:
    return trainer_command if any(ch in trainer_command for ch in "|&;<>()$`") else shlex.split(trainer_command)


def maybe_trigger_online_rl_training(
    db,
    *,
    runtime_base_url: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Export a pending batch and spawn the local MIS-PO trainer hook when configured."""
    cfg = cfg or load_online_rl_config()
    if not online_rl_enabled_for_runtime(runtime_base_url=runtime_base_url, cfg=cfg):
        return None

    rows = db.build_rl_feedback_export_rows(
        pending_only=True,
        include_neutral=False,
        limit=int(cfg.get("max_batch_size", 64)),
    )
    if len(rows) < int(cfg.get("min_batch_size", 8)):
        return None

    export_dir = Path(str(cfg.get("export_dir") or "")).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    export_path = export_dir / f"feedback_batch_{stamp}_{len(rows)}.jsonl"
    db.export_rl_feedback_jsonl(
        export_path,
        pending_only=True,
        include_neutral=False,
        limit=int(cfg.get("max_batch_size", 64)),
    )

    feedback_ids = [int(row["feedback_id"]) for row in rows]
    db.mark_rl_feedback_status(
        feedback_ids,
        trainer_status="queued",
        export_path=str(export_path),
        last_error=None,
    )

    trainer_command = str(cfg.get("trainer_command") or "").strip()
    invocation: str | list[str] | None
    if trainer_command:
        invocation = _configured_trainer_invocation(trainer_command)
    elif cfg.get("builtin_trainer", True):
        invocation = _builtin_trainer_invocation(
            export_path,
            runtime_base_url=runtime_base_url or str(cfg.get("model_base_url") or ""),
        )
    else:
        invocation = None

    if not invocation:
        return None

    env = os.environ.copy()
    env.update(
        {
            "HERMES_RL_EXPORT_PATH": str(export_path),
            "HERMES_RL_BATCH_SIZE": str(len(rows)),
            "HERMES_RL_FEEDBACK_IDS": ",".join(str(fid) for fid in feedback_ids),
            "HERMES_RL_RUNTIME_BASE_URL": str(runtime_base_url or cfg.get("model_base_url") or ""),
        }
    )

    subprocess.Popen(
        invocation,
        shell=isinstance(invocation, str),
        cwd=str(export_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "queued",
        "batch_size": len(rows),
        "export_path": str(export_path),
        "feedback_ids": feedback_ids,
        "algorithm": str(cfg.get("algorithm") or "mis_po"),
    }


def submit_online_rl_feedback(
    db,
    *,
    session_id: str,
    message_id: int,
    label: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    runtime_base_url: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Persist a feedback label and optionally queue a local trainer batch."""
    cfg = cfg or load_online_rl_config()
    normalized = normalize_rl_feedback_label(label)

    runtime_state = load_online_rl_state(cfg)
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("online_rl_algorithm", str(cfg.get("algorithm") or "mis_po"))
    merged_metadata.setdefault(
        "online_rl_backend",
        resolve_online_rl_backend(runtime_base_url=runtime_base_url, cfg=cfg),
    )
    if runtime_state.get("active_model_name"):
        merged_metadata.setdefault("online_rl_model_name", runtime_state.get("active_model_name"))
    if runtime_state.get("active_adapter_path"):
        merged_metadata.setdefault("online_rl_adapter_path", runtime_state.get("active_adapter_path"))

    feedback = db.set_rl_feedback(
        session_id=session_id,
        message_id=message_id,
        label=normalized,
        source=source,
        metadata=merged_metadata or None,
    )

    queued = None
    if is_trainable_rl_feedback_label(normalized):
        queued = maybe_trigger_online_rl_training(
            db,
            runtime_base_url=runtime_base_url,
            cfg=cfg,
        )
    return feedback, queued


def feedback_choice_metadata() -> list[Dict[str, Any]]:
    """Return stable frontend metadata for the supported feedback labels."""
    return [
        {
            "label": RL_FEEDBACK_UPWEIGHT,
            "title": "RL upweight",
            "reward": 1.0,
            "description": "Positive binary reward for the full assistant trajectory.",
        },
        {
            "label": RL_FEEDBACK_DOWNWEIGHT,
            "title": "RL downweight",
            "reward": -1.0,
            "description": "Negative binary reward for the full assistant trajectory.",
        },
        {
            "label": RL_FEEDBACK_NO_RL,
            "title": "No RL",
            "reward": 0.0,
            "description": "Store no trainable reward for this turn.",
        },
    ]

