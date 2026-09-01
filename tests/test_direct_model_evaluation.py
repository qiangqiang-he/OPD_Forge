"""CPU-only tests for direct local-model evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import utils.evaluate_checkpoints as checkpoint_evaluation
import utils.evaluate_models as model_evaluation
from utils.evaluate_checkpoints import (
    DIRECT_MODEL_SOURCE,
    CheckpointSpec,
    evaluate_checkpoint,
    is_huggingface_model_dir,
    load_evaluation_config,
    load_tasks,
)
from utils.evaluate_models import (
    build_model_validation_plan,
    model_result_path,
    summary_result_path,
)


def _write_fake_hf_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "test", "max_position_embeddings": 128}),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"weights")


def _write_direct_config(
    project_root: Path,
    *,
    gpus: list[int] | None = None,
    rollouts_per_gpu_batch: int = 256,
) -> Path:
    data_dir = project_root / "data"
    config_dir = project_root / "configs" / "eval"
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (data_dir / "tiny.json").write_text(
        json.dumps([{"question": "1+1?", "answer": "2"}]),
        encoding="utf-8",
    )
    for name in ("model-a", "model-b"):
        _write_fake_hf_model(project_root / "models" / name)

    config_path = config_dir / "models.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "run_name": "direct-model-test",
                "model_source": {
                    "type": "hf_models",
                    "models": [
                        {"name": "model-a", "path": "./models/model-a"},
                        {"name": "model-b", "path": "./models/model-b"},
                    ],
                },
                "datasets": [{"name": "tiny", "path": "./data/tiny.json"}],
                "prompt": "qwen3_no_thinking_prompt",
                "seed": 42,
                "sampling": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "n": 16,
                    "max_new_tokens": 8,
                },
                "length_control": [4, 8],
                "report": {
                    "metrics": [
                        "avg_at_n",
                        "pass_at_n",
                        "mean_length",
                        "truncation_rate",
                    ]
                },
                "runtime": {
                    "gpus": list(range(8)) if gpus is None else gpus,
                    "tensor_parallel_size": 1,
                    "rollouts_per_gpu_batch": rollouts_per_gpu_batch,
                },
                "engine": {
                    "dtype": "bfloat16",
                    "trust_remote_code": True,
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 16,
                    "enable_prefix_caching": True,
                },
                "result_dir": "./eval_results",
            }
        ),
        config_path,
    )
    return config_path


def test_direct_config_resolves_models_and_enforces_full_replica_runtime(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    config_path = _write_direct_config(project_root)

    config = load_evaluation_config(
        config_path,
        project_root=project_root,
        source_type=DIRECT_MODEL_SOURCE,
    )
    tasks, counts = load_tasks(config)

    assert config.training_output is None
    assert [model.name for model in config.models] == ["model-a", "model-b"]
    assert all(is_huggingface_model_dir(model.path) for model in config.models)
    assert config.gpus == tuple(range(8))
    assert config.tensor_parallel_size == 1
    assert config.rollouts_per_gpu_batch == 256
    assert config.questions_per_gpu_batch == 16
    assert counts == {"tiny": 1}
    assert len(tasks) == 1
    assert "vllm" not in sys.modules


@pytest.mark.parametrize(
    ("gpus", "rollouts", "message"),
    [
        (list(range(7)), 256, "exactly eight"),
        (list(range(8)), 128, "exactly 256"),
    ],
)
def test_direct_config_rejects_non_server_runtime(
    tmp_path: Path,
    gpus: list[int],
    rollouts: int,
    message: str,
):
    project_root = tmp_path / "project"
    config_path = _write_direct_config(
        project_root,
        gpus=gpus,
        rollouts_per_gpu_batch=rollouts,
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_config(
            config_path,
            project_root=project_root,
            source_type=DIRECT_MODEL_SOURCE,
        )


def test_sharded_model_with_missing_indexed_weight_is_rejected(tmp_path: Path):
    model_dir = tmp_path / "incomplete-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert not is_huggingface_model_dir(model_dir)


def test_validation_plan_records_each_model_and_fixed_batch_contract(tmp_path: Path):
    project_root = tmp_path / "project"
    config = load_evaluation_config(
        _write_direct_config(project_root),
        project_root=project_root,
        source_type=DIRECT_MODEL_SOURCE,
    )
    tasks, counts = load_tasks(config)
    plan = build_model_validation_plan(
        config,
        tasks=tasks,
        dataset_question_counts=counts,
        max_prompt_tokens_by_model={"model-a": 5, "model-b": 6},
    )

    assert [model["name"] for model in plan["models"]] == ["model-a", "model-b"]
    assert [model["max_prompt_tokens"] for model in plan["models"]] == [5, 6]
    assert plan["runtime"] == {
        "gpus": list(range(8)),
        "tensor_parallel_size": 1,
        "rollouts_per_gpu_batch": 256,
        "questions_per_gpu_batch": 16,
        "full_model_replica_per_gpu": True,
    }
    assert plan["batches_per_model"] == 1
    assert model_result_path(config, config.models[0]) == (
        project_root / "eval_results/direct-model-test/model-a.json"
    )
    assert summary_result_path(config) == (
        project_root / "eval_results/direct-model-test/summary.json"
    )


def test_checkpoint_entry_remains_a_wrapper_around_shared_model_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    observed: dict[str, object] = {}

    def fake_evaluate_model(**kwargs):
        observed.update(kwargs)
        return {"tiny": {}}

    monkeypatch.setattr(checkpoint_evaluation, "evaluate_model", fake_evaluate_model)
    checkpoint = CheckpointSpec(
        step_num=20,
        checkpoint_dir=tmp_path / "global_step_20",
        source_kind="huggingface",
        source_dir=tmp_path / "model",
        tokenizer_dir=tmp_path / "model",
        model_dir=tmp_path / "model",
    )
    config = SimpleNamespace(run_name="run")

    result = evaluate_checkpoint(
        config=config,
        checkpoint=checkpoint,
        model_dir=tmp_path / "model",
        batches=[[{"prompt": "p"}]],
        dataset_question_counts={"tiny": 1},
    )

    assert result == {"tiny": {}}
    assert observed["model_dir"] == tmp_path / "model"
    assert observed["progress_label"] == "step=20"
    assert observed["process_name_prefix"] == "eval-step20"
    assert observed["start_event"] == "STEP_START"


def test_direct_launcher_calls_only_the_direct_model_entrypoint():
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" / "start_model_eval.sh"
    ).read_text(encoding="utf-8")

    assert "python -m utils.evaluate_models" in launcher
    assert "python -m utils.evaluate_checkpoints" not in launcher


def test_completed_model_result_survives_later_failure_and_models_are_never_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    project_root = tmp_path / "project"
    config_path = _write_direct_config(project_root)
    model_a = project_root / "models/model-a"
    model_b = project_root / "models/model-b"

    monkeypatch.setattr(model_evaluation, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        model_evaluation,
        "validate_prompt_budget",
        lambda config, tasks, model_path: 5,
    )

    def fake_evaluate_model(*, model_dir: Path, **kwargs):
        del kwargs
        if model_dir == model_b:
            raise RuntimeError("simulated second-model failure")
        return {"tiny": {"8": {"Avg@16": 1.0}}}

    monkeypatch.setattr(model_evaluation, "evaluate_model", fake_evaluate_model)

    with pytest.raises(RuntimeError, match="simulated second-model failure"):
        model_evaluation.main(["--config", str(config_path)])

    first_result = project_root / "eval_results/direct-model-test/model-a.json"
    summary_path = project_root / "eval_results/direct-model-test/summary.json"
    assert first_result.is_file()
    assert json.loads(first_result.read_text(encoding="utf-8"))["model_name"] == "model-a"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert [item["status"] for item in summary["models"]] == [
        "completed",
        "failed",
    ]
    assert model_a.is_dir()
    assert model_b.is_dir()
    assert "vllm" not in sys.modules

    with pytest.raises(FileExistsError, match="Use a new run_name"):
        model_evaluation.main(["--config", str(config_path)])
