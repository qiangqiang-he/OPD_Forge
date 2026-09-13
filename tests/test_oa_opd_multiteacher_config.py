"""CPU-only checks for the user-editable OA--OPD Teacher mapping.

The large vLLM runner is intentionally not started here.  These tests keep
the configuration layer honest: a run must resolve a stable, filesystem-safe
Teacher key, preserve all configured Teachers, and bind the complete mapping
to its resumability signature.  They run in the local WSL environment without
loading either model.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tests import run_oa_opd_h20_avg128 as runner


def test_configured_teacher_mapping_normalizes_paths_without_touching_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "teacher-a"
    second = tmp_path / "teacher-b"
    monkeypatch.setattr(
        runner,
        "TEACHER_MODELS",
        {"teacher_a": str(first), "teacher_b": second},
    )

    resolved = runner.configured_teacher_models()

    assert resolved == {"teacher_a": first, "teacher_b": second}
    # The helper returns a fresh mapping; callers cannot accidentally mutate
    # the module-level configuration while constructing a run plan.
    resolved["teacher_a"] = tmp_path / "changed"
    assert runner.TEACHER_MODELS["teacher_a"] == str(first)


@pytest.mark.parametrize("bad_key", ["Teacher-A", "../escape", "", "a/b", "a b"])
def test_configured_teacher_mapping_rejects_unsafe_keys(
    monkeypatch: pytest.MonkeyPatch, bad_key: str
) -> None:
    monkeypatch.setattr(runner, "TEACHER_MODELS", {bad_key: "model"})
    with pytest.raises(ValueError, match="Teacher key"):
        runner.configured_teacher_models()


def test_configured_teacher_mapping_rejects_empty_or_non_path_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "TEACHER_MODELS", {})
    with pytest.raises(ValueError, match="non-empty dict"):
        runner.configured_teacher_models()

    monkeypatch.setattr(runner, "TEACHER_MODELS", {"teacher": object()})
    with pytest.raises(TypeError, match="model path"):
        runner.configured_teacher_models()


def test_resolve_teacher_models_uses_all_configured_teachers_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = {
        "qwen3_4b_instruct_2507": tmp_path / "instruct",
        "qwen3_4b": tmp_path / "base",
    }
    monkeypatch.setattr(runner, "TEACHER_MODELS", configured)

    resolved = runner.resolve_teacher_models(SimpleNamespace())

    assert resolved == configured
    assert list(resolved) == ["qwen3_4b_instruct_2507", "qwen3_4b"]


def test_resolve_teacher_models_keeps_legacy_single_teacher_override(
    tmp_path: Path,
) -> None:
    model = tmp_path / "Qwen3-4B"
    resolved = runner.resolve_teacher_models(SimpleNamespace(teacher_model=model))

    assert resolved == {"qwen3-4b": model}


def test_resolve_teacher_models_can_select_a_configured_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner,
        "TEACHER_MODELS",
        {"teacher_a": tmp_path / "a", "teacher_b": tmp_path / "b"},
    )
    selected = runner.resolve_teacher_models(
        SimpleNamespace(teacher_key=["teacher_b"])
    )
    assert selected == {"teacher_b": tmp_path / "b"}


def test_multiteacher_signature_binds_complete_mapping_and_is_order_independent(
    tmp_path: Path,
) -> None:
    student = tmp_path / "student"
    teachers_a = {
        "teacher_a": tmp_path / "teacher-a",
        "teacher_b": tmp_path / "teacher-b",
    }
    teachers_b = {
        "teacher_b": teachers_a["teacher_b"],
        "teacher_a": teachers_a["teacher_a"],
    }

    signature_a = runner.run_signature(
        "dataset-sha", "correct", student, teacher_models=teachers_a, world_size=8
    )
    signature_b = runner.run_signature(
        "dataset-sha", "correct", student, teacher_models=teachers_b, world_size=8
    )
    signature_changed = runner.run_signature(
        "dataset-sha",
        "correct",
        student,
        teacher_models={**teachers_a, "teacher_c": tmp_path / "teacher-c"},
        world_size=8,
    )

    assert signature_a == signature_b
    assert signature_a != signature_changed


def test_run_signature_requires_a_teacher_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires teacher"):
        runner.run_signature("dataset-sha", "correct", tmp_path / "student")


def test_teacher_specific_shards_are_namespaced_while_keep_delete_are_shared(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    proposal_a = runner._proposal_paths(root, 2, 7, "teacher_a")
    proposal_b = runner._proposal_paths(root, 2, 7, "teacher_b")
    replace_a = runner._arm_paths(root, 2, "replace", 7, "teacher_a")
    replace_b = runner._arm_paths(root, 2, "replace", 7, "teacher_b")
    keep = runner._arm_paths(root, 2, "keep", 7)
    delete = runner._arm_paths(root, 2, "delete", 7)

    assert proposal_a != proposal_b
    assert proposal_a.parts[-2] == "teacher_a"
    assert proposal_b.parts[-2] == "teacher_b"
    assert replace_a != replace_b
    assert replace_a.parts[-2] == "teacher_a"
    assert replace_b.parts[-2] == "teacher_b"
    assert keep.parts[-2:] == ("keep", "part_00007.json.gz")
    assert delete.parts[-2:] == ("delete", "part_00007.json.gz")

    with pytest.raises(ValueError, match="Teacher key"):
        runner._proposal_paths(root, 0, 0, "../escape")


def test_multi_teacher_continuations_share_student_and_separate_replace_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keyed orchestration must produce one Replace shard per Teacher.

    This is a CPU-only mock: it exercises exact-prefix construction, metadata
    binding, and shard validation without loading vLLM or either model.
    """

    from tests.test_run_oa_opd_h20_avg128 import _dataset

    _, cases = runner.load_dataset(_dataset(tmp_path), "correct")
    batches = runner.shard_plan(cases)
    root = tmp_path / "multi-out"
    args = SimpleNamespace(
        outcome="correct",
        worker_id=0,
        world_size=1,
        gpu_id=0,
        validate_only=False,
        overwrite=False,
        min_free_gpu_gib=5.0,
        gpu_memory_utilization=0.5,
        heartbeat_seconds=30.0,
        memory_check_seconds=2.0,
        student_model=Path("student"),
        arm="all",
        run_id="mock",
    )

    class _Tok:
        def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            del skip_special_tokens, clean_up_tokenization_spaces
            return "A" if list(token_ids) == [65] else "B"

    monkeypatch.setattr(runner, "_load_tokenizer", lambda _: _Tok())
    llm_calls = []
    monkeypatch.setattr(
        runner,
        "_new_llm",
        lambda *call_args: (llm_calls.append(call_args) or object()),
    )
    monkeypatch.setattr(runner, "_check_headroom", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_completion_item",
        lambda *, sample_index, **_: {
            "sample_index": sample_index,
            "continuation_token_ids": [sample_index],
            "continuation_num_tokens": 1,
            "continuation_token_ids_sha256": runner.token_hash([sample_index]),
            "full_response_num_tokens": 1,
            "full_response_token_ids_sha256": runner.token_hash([sample_index]),
            "full_response_text": "A",
            "full_response_text_sha256": runner.text_hash("A"),
            "finish_reason": "stop",
            "stop_reason": None,
            "correct": 1,
            "format_valid": 1,
            "extracted_answer": "1",
            "reward": 1,
            "truncated_by_response_cap": False,
        },
    )

    def fake_generate(_llm, prompts, _params, _args, _stage):
        return [
            SimpleNamespace(
                prompt_token_ids=prompt,
                outputs=[SimpleNamespace(index=i) for i in range(runner.SAMPLES_PER_ARM)],
            )
            for prompt in prompts
        ]

    monkeypatch.setattr(runner, "_generate", fake_generate)

    def proposal_for(key: str, token: int) -> dict[str, dict[str, object]]:
        return {
            str(case["case_id"]): {
                "case_id": case["case_id"],
                "case_index": case["case_index"],
                "record_index": case["record_index"],
                "replacement_step_token_ids": [token],
                "replacement_step_token_ids_sha256": runner.token_hash([token]),
                "replacement_step_decoded_text": "A" if token == 65 else "B",
                "teacher_model": f"/models/{key}",
                "teacher_key": key,
                "request_seed": 1,
            }
            for case in cases
        }

    proposals = {
        "teacher_a": (proposal_for("teacher_a", 65), {}),
        "teacher_b": (proposal_for("teacher_b", 66), {}),
    }
    runner.run_continuations(
        args,
        root,
        cases,
        batches,
        "sig",
        runner.derive_max_model_len(cases),
        {},
        {},
        teacher_proposals=proposals,
    )

    # One Student instance serves all four logical arms.
    assert len(llm_calls) == 1
    for arm in ("keep", "delete"):
        path = runner._arm_paths(root, 0, arm, 0)
        assert path.is_file()
        assert runner._valid_arm_shard(path, batches[0], "sig", arm)
    for key, token in (("teacher_a", 65), ("teacher_b", 66)):
        path = runner._arm_paths(root, 0, "replace", 0, key)
        assert path.is_file()
        payload = runner.load_json_maybe_gzip(path)
        assert all(row["teacher_key"] == key for row in payload["records"])
        assert all(row["response_prefix_token_ids"][-1] == token for row in payload["records"])
        assert runner._valid_arm_shard(
            path, batches[0], "sig", "replace", proposals[key][0], teacher_key=key
        )


def test_main_orchestrates_all_configured_teachers_without_reloading_student(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny end-to-end mock catches wiring errors in the CLI entry point."""

    from tests.test_run_oa_opd_h20_avg128 import _dataset

    dataset = _dataset(tmp_path)
    student = tmp_path / "student"
    teacher_a = tmp_path / "teacher-a"
    teacher_b = tmp_path / "teacher-b"
    student.mkdir()
    teacher_a.mkdir()
    teacher_b.mkdir()
    monkeypatch.setattr(
        runner,
        "TEACHER_MODELS",
        {"teacher_a": teacher_a, "teacher_b": teacher_b},
    )

    class _Tok:
        def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            del skip_special_tokens, clean_up_tokenization_spaces
            values = list(token_ids)
            if values == [65]:
                return "A"
            if values == [66]:
                return "B"
            return "A"

    monkeypatch.setattr(runner, "_load_tokenizer", lambda _: _Tok())
    monkeypatch.setattr(runner, "_check_headroom", lambda *args: {})
    llm_models = []
    monkeypatch.setattr(
        runner,
        "_new_llm",
        lambda model, *args: (llm_models.append(Path(model)) or object()),
    )
    monkeypatch.setattr(runner.base, "shutdown_vllm", lambda _: None)

    def fake_proposal_record(case, _output, _tokenizer, _seed, _attempt, model, key=None):
        token = 65 if key == "teacher_a" else 66
        return {
            "case_id": case["case_id"],
            "case_index": case["case_index"],
            "record_index": case["record_index"],
            "replacement_step_token_ids": [token],
            "replacement_step_token_ids_sha256": runner.token_hash([token]),
            "replacement_step_decoded_text": "A" if token == 65 else "B",
            "teacher_model": str(model),
            "teacher_key": key,
            "request_seed": 1,
        }

    monkeypatch.setattr(runner, "_proposal_record", fake_proposal_record)
    monkeypatch.setattr(
        runner,
        "_completion_item",
        lambda *, sample_index, **_: {
            "sample_index": sample_index,
            "continuation_token_ids": [sample_index],
            "continuation_num_tokens": 1,
            "continuation_token_ids_sha256": runner.token_hash([sample_index]),
            "full_response_num_tokens": 1,
            "full_response_token_ids_sha256": runner.token_hash([sample_index]),
            "full_response_text": "A",
            "full_response_text_sha256": runner.text_hash("A"),
            "finish_reason": "stop",
            "stop_reason": None,
            "correct": 1,
            "format_valid": 1,
            "extracted_answer": "1",
            "reward": 1,
            "truncated_by_response_cap": False,
        },
    )

    def fake_generate(_llm, prompts, _params, _args, _stage):
        result = []
        for prompt, params in zip(prompts, _params, strict=True):
            # Proposal SamplingParams has n=1; continuation has n=128.
            n = int(getattr(params, "n", 1))
            if n == 1:
                result.append(
                    SimpleNamespace(
                        prompt_token_ids=prompt,
                        outputs=[SimpleNamespace(token_ids=[65], finish_reason="stop", stop_reason=None)],
                    )
                )
            else:
                result.append(
                    SimpleNamespace(
                        prompt_token_ids=prompt,
                        outputs=[SimpleNamespace(index=i) for i in range(n)],
                    )
                )
        return result

    monkeypatch.setattr(runner, "_generate", fake_generate)
    output_root = tmp_path / "output"
    argv = [
        "runner",
        "--outcome",
        "correct",
        "--dataset",
        str(dataset),
        "--output-root",
        str(output_root),
        "--student-model",
        str(student),
        "--world-size",
        "1",
        "--worker-id",
        "0",
        "--gpu-id",
        "0",
        "--phase",
        "all",
        "--arm",
        "all",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()

    run_dir = output_root / "correct"
    manifest = runner.load_json(run_dir / "run_manifest.json")
    assert list(manifest["teacher_models"]) == ["teacher_a", "teacher_b"]
    assert len(llm_models) == 3  # Teacher A, Teacher B, then one Student.
    assert runner._proposal_paths(run_dir, 0, 0, "teacher_a").is_file()
    assert runner._proposal_paths(run_dir, 0, 0, "teacher_b").is_file()
    assert runner._arm_paths(run_dir, 0, "keep", 0).is_file()
    assert runner._arm_paths(run_dir, 0, "delete", 0).is_file()
    assert runner._arm_paths(run_dir, 0, "replace", 0, "teacher_a").is_file()
    assert runner._arm_paths(run_dir, 0, "replace", 0, "teacher_b").is_file()
