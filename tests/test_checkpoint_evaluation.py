"""CPU-only tests for the checkpoint evaluation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from omegaconf import OmegaConf

from utils.evaluate_checkpoints import (
    PROJECT_ROOT,
    CheckpointSpec,
    aggregate_question_results,
    atomic_write_json,
    discover_checkpoints,
    load_evaluation_config,
    prepare_model,
    score_completion_at_lengths,
)


def test_repository_eval_config_keeps_the_formal_eight_gpu_contract_without_vllm():
    raw = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs/eval/eval.yaml"), resolve=True
    )

    assert raw["runtime"] == {
        "gpus": list(range(8)),
        "tensor_parallel_size": 1,
        "rollouts_per_gpu_batch": 256,
    }
    assert raw["sampling"]["n"] == 16
    assert raw["sampling"]["max_new_tokens"] == 32768
    assert raw["length_control"][-1] == raw["sampling"]["max_new_tokens"]
    assert "vllm" not in sys.modules


def test_config_derives_run_name_and_sorts_numeric_checkpoint_steps(tmp_path: Path):
    project_root = tmp_path / "project"
    training_output = project_root / "outputs" / "example_run"
    data_dir = project_root / "data"
    config_dir = project_root / "configs" / "eval"
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (data_dir / "tiny.json").write_text(
        json.dumps([{"question": "1+1?", "answer": "2"}]), encoding="utf-8"
    )
    for step in (100, 20, 40):
        actor = training_output / f"global_step_{step}" / "actor"
        (actor / "huggingface").mkdir(parents=True)
        (actor / "fsdp_config.json").write_text(
            json.dumps({"world_size": 2}), encoding="utf-8"
        )
        (actor / "huggingface" / "config.json").write_text("{}", encoding="utf-8")
        for rank in range(2):
            (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(b"shard")

    config_path = config_dir / "tiny.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "training_output": "./outputs/example_run",
                "datasets": [{"name": "tiny", "path": "./data/tiny.json"}],
                "prompt": "qwen3_no_thinking_prompt",
                "seed": 42,
                "sampling": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "n": 2,
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
                    "gpus": [0, 1],
                    "tensor_parallel_size": 1,
                    "rollouts_per_gpu_batch": 4,
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

    config = load_evaluation_config(config_path, project_root=project_root)
    checkpoints = discover_checkpoints(config)

    assert config.run_name == "example_run"
    assert [checkpoint.step_num for checkpoint in checkpoints] == [20, 40, 100]
    assert [checkpoint.source_dir.name for checkpoint in checkpoints] == [
        "actor",
        "actor",
        "actor",
    ]


def test_posthoc_length_scoring_uses_token_prefix_and_tracks_truncation():
    decoded = {2: "wrong", 4: "correct"}
    values = score_completion_at_lengths(
        token_ids=[10, 11, 12, 13],
        full_text="correct",
        finish_reason="stop",
        answer="gold",
        length_control=[2, 4],
        decode=lambda ids: decoded[len(ids)],
        verify=lambda response, answer: float(response == "correct" and answer == "gold"),
    )

    assert values == {
        2: {"correct": 0, "tokens": 2, "truncated": 1},
        4: {"correct": 1, "tokens": 4, "truncated": 0},
    }

    length_finished = score_completion_at_lengths(
        token_ids=[10, 11, 12, 13],
        full_text="correct",
        finish_reason="length",
        answer="gold",
        length_control=[4],
        decode=lambda ids: "unused",
        verify=lambda response, answer: 1.0,
    )
    assert length_finished[4]["truncated"] == 1


def test_aggregate_emits_exactly_the_four_requested_metrics():
    records = [
        {
            "dataset": "math",
            "question_index": 0,
            "lengths": {
                "2": {
                    "rollouts": 2,
                    "correct_count": 1,
                    "token_count": 4,
                    "truncated_count": 2,
                },
                "4": {
                    "rollouts": 2,
                    "correct_count": 2,
                    "token_count": 7,
                    "truncated_count": 1,
                },
            },
        },
        {
            "dataset": "math",
            "question_index": 1,
            "lengths": {
                "2": {
                    "rollouts": 2,
                    "correct_count": 0,
                    "token_count": 3,
                    "truncated_count": 1,
                },
                "4": {
                    "rollouts": 2,
                    "correct_count": 1,
                    "token_count": 5,
                    "truncated_count": 0,
                },
            },
        },
    ]

    result = aggregate_question_results(
        question_results=records,
        dataset_question_counts={"math": 2},
        dataset_order=["math"],
        length_control=[2, 4],
        n=2,
    )

    assert result["math"]["2"] == {
        "Avg@2": 0.25,
        "Pass@2": 0.5,
        "mean_length": 1.75,
        "truncation_rate": 0.75,
    }
    assert result["math"]["4"] == {
        "Avg@2": 0.75,
        "Pass@2": 1.0,
        "mean_length": 3.0,
        "truncation_rate": 0.25,
    }
    assert set(result["math"]["4"]) == {
        "Avg@2",
        "Pass@2",
        "mean_length",
        "truncation_rate",
    }


def test_atomic_json_is_utf8_unescaped_and_four_space_indented(tmp_path: Path):
    output = tmp_path / "result.json"
    atomic_write_json(output, {"名称": "评测", "steps": [{"step_num": 20}]})

    raw = output.read_bytes()
    text = raw.decode("utf-8")
    assert '"名称": "评测"' in text
    assert '\\u8bc4' not in text
    assert '\n    "名称"' in text
    assert '\n        {' in text


def test_fsdp_merge_lives_under_training_output_and_is_removed_on_success_or_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_root = tmp_path / "project"
    training_output = project_root / "outputs" / "example_run"
    source_dir = training_output / "global_step_20" / "actor"
    result_dir = project_root / "eval_results"
    source_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    config = SimpleNamespace(
        project_root=project_root,
        training_output=training_output,
        result_dir=result_dir,
    )
    checkpoint = CheckpointSpec(
        step_num=20,
        checkpoint_dir=source_dir.parent,
        source_kind="fsdp",
        source_dir=source_dir,
        tokenizer_dir=source_dir / "huggingface",
        model_dir=None,
    )
    observed: list[Path] = []

    def fake_merge(command, *, cwd, env, check):
        del cwd, env, check
        target = Path(command[command.index("--target_dir") + 1])
        observed.append(target)
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr("utils.evaluate_checkpoints.subprocess.run", fake_merge)

    with prepare_model(config, checkpoint) as model_dir:
        assert model_dir == observed[-1]
        assert training_output in model_dir.parents
        assert result_dir not in model_dir.parents
        assert model_dir.is_dir()

    assert not observed[-1].parent.exists()

    with pytest.raises(RuntimeError, match="simulated evaluation failure"):
        with prepare_model(config, checkpoint) as model_dir:
            assert model_dir == observed[-1]
            assert training_output in model_dir.parents
            assert result_dir not in model_dir.parents
            assert model_dir.is_dir()
            raise RuntimeError("simulated evaluation failure")

    assert len(observed) == 2
    assert not observed[-1].parent.exists()
    assert not list(training_output.glob(".eval_merged_global_step_*"))
    assert not list(result_dir.iterdir())
