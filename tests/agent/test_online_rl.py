"""Tests for agent/online_rl.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.online_rl import (
    maybe_trigger_online_rl_training,
    resolve_online_rl_model,
    save_online_rl_state,
    submit_online_rl_feedback,
)
from hermes_state import SessionDB


def _cfg(tmp_path: Path, **overrides):
    cfg = {
        "enabled": True,
        "prompt_after_response": True,
        "local_only": True,
        "backend": "vllm",
        "algorithm": "mis_po",
        "min_batch_size": 2,
        "max_batch_size": 8,
        "feedback_timeout_seconds": 30,
        "export_dir": str(tmp_path / "exports"),
        "adapter_output_dir": str(tmp_path / "adapters"),
        "state_path": str(tmp_path / "state.json"),
        "trainer_command": "",
        "builtin_trainer": True,
        "model_base_url": "http://localhost:8000/v1",
        "training_base_model": "local/test-model",
        "adapter_name": "hermes-online-rl",
        "ollama_model_name": "",
    }
    cfg.update(overrides)
    return cfg


def _db_with_feedback_targets(tmp_path: Path) -> tuple[SessionDB, str, int, int]:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess_rl"
    db.create_session(session_id=session_id, source="cli", model="local/test-model")
    db.append_message(session_id, role="user", content="Question one")
    first_id = db.append_message(session_id, role="assistant", content="Answer one")
    db.append_message(session_id, role="user", content="Question two")
    second_id = db.append_message(session_id, role="assistant", content="Answer two")
    return db, session_id, first_id, second_id


class TestSubmitOnlineRLFeedback:
    def test_submit_no_rl_persists_metadata_without_queueing(self, tmp_path):
        db, session_id, message_id, _ = _db_with_feedback_targets(tmp_path)
        try:
            feedback, queued = submit_online_rl_feedback(
                db,
                session_id=session_id,
                message_id=message_id,
                label="no_rl",
                source="cli",
                metadata={"interactive": True},
                runtime_base_url="http://localhost:8000/v1",
                cfg=_cfg(tmp_path),
            )
            assert queued is None
            assert feedback["label"] == "no_rl"
            assert feedback["reward"] == 0.0
            assert feedback["metadata"]["interactive"] is True
            assert feedback["metadata"]["online_rl_algorithm"] == "mis_po"
            assert feedback["metadata"]["online_rl_backend"] == "vllm"
        finally:
            db.close()


class TestQueueTraining:
    def test_maybe_trigger_training_exports_and_marks_rows_queued(self, tmp_path):
        db, session_id, first_id, second_id = _db_with_feedback_targets(tmp_path)
        cfg = _cfg(tmp_path)
        try:
            db.set_rl_feedback(session_id, first_id, "upweight", source="cli")
            db.set_rl_feedback(session_id, second_id, "downweight", source="cli")

            with patch("agent.online_rl.subprocess.Popen") as popen:
                popen.return_value = MagicMock()
                queued = maybe_trigger_online_rl_training(
                    db,
                    runtime_base_url="http://localhost:8000/v1",
                    cfg=cfg,
                )

            assert queued is not None
            assert queued["status"] == "queued"
            assert queued["batch_size"] == 2
            assert Path(queued["export_path"]).exists()

            rows = db.list_rl_feedback(limit=10, pending_only=False, include_neutral=True)
            assert len(rows) == 2
            assert {row["trainer_status"] for row in rows} == {"queued"}
            popen.assert_called_once()
        finally:
            db.close()


class TestResolveOnlineRLModel:
    def test_resolve_online_rl_model_activates_vllm_adapter(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter_dir = tmp_path / "adapters" / "adapter_00001"
        adapter_dir.mkdir(parents=True)
        save_online_rl_state(
            {
                "backend": "vllm",
                "active_model_name": "hermes-online-rl",
                "active_adapter_path": str(adapter_dir),
                "base_model_name": "local/test-model",
            },
            cfg,
        )

        with patch("agent.online_rl._load_vllm_adapter") as load_adapter:
            resolved = resolve_online_rl_model(
                "local/test-model",
                runtime_base_url="http://localhost:8000/v1",
                cfg=cfg,
            )

        assert resolved["active"] is True
        assert resolved["model"] == "hermes-online-rl"
        load_adapter.assert_called_once_with(
            runtime_base_url="http://localhost:8000/v1",
            adapter_name="hermes-online-rl",
            adapter_path=str(adapter_dir),
        )
