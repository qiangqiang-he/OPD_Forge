from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import run_oa_opd_h20_avg128 as runner


def _dataset(tmp_path: Path, outcome: str = "correct") -> Path:
    prompt = [101, 102]
    response_ids = list(range(20))
    source = "".join(chr(65 + i) for i in range(20))
    steps = []
    selected = []
    for i in range(5):
        start, end = i * 4, (i + 1) * 4
        text = source[start:end]
        step = {
            "step_index": i,
            "reasoning_step_index": i,
            "char_start": start,
            "char_end": end,
            "token_start": start,
            "token_end": end,
            "num_tokens": 4,
            "text": text,
        }
        steps.append(step)
        selected.append(
            {
                "case_id": f"7:{i}",
                "cohort_case_index": i,
                "global_case_index": i,
                "response_position_bin": f"p{i + 1}",
                "response_position_bin_index": i,
                "response_position_fraction": i / 4,
                "response_position_third": "early" if i < 2 else ("late" if i >= 3 else "middle"),
                "step_offset": i,
                "step_index": i,
                "reasoning_step_index": i,
                "step_char_start": start,
                "step_char_end": end,
                "step_token_start": start,
                "step_token_end": end,
                "step_num_tokens": 4,
                "step_text_sha256": runner.text_hash(text),
                "step_token_ids_sha256": runner.token_hash(response_ids[start:end]),
            }
        )
    row = {
        "schema_version": 1,
        "experiment": runner.EXPERIMENT,
        "trajectory_id": f"{outcome}:7",
        "record_index": 7,
        "source_outcome": outcome,
        "outcome_response_rank": 1,
        "question": "q",
        "answer": "1",
        "source_response": source,
        "source_extracted_answer": "1" if outcome == "correct" else "0",
        "source_correct": 1 if outcome == "correct" else 0,
        "source_format_valid": 1,
        "source_finish_reason": "stop",
        "source_response_text_sha256": runner.text_hash(source),
        "prompt_token_ids": prompt,
        "prompt_num_tokens": len(prompt),
        "prompt_token_ids_sha256": runner.token_hash(prompt),
        "source_response_token_ids": response_ids,
        "source_response_num_tokens": len(response_ids),
        "source_response_token_ids_sha256": runner.token_hash(response_ids),
        "step_partition": {"steps": steps, "boundary_probes": [{"response_token_end": i * 4} for i in range(6)]},
        "selected_steps": selected,
    }
    payload = {
        "metadata": {
            "source_outcome": outcome,
            "response_count": 1,
            "selected_step_count": 5,
            "steps_per_response": 5,
            "samples_per_arm": 128,
            "max_response_tokens": 8192,
            "teacher_proposal_max_new_tokens": 512,
        },
        "responses": [row],
    }
    path = tmp_path / f"{outcome}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_nested_loader_preserves_exact_token_slices(tmp_path: Path) -> None:
    metadata, cases = runner.load_dataset(_dataset(tmp_path), "correct")
    assert metadata["samples_per_arm"] == 128
    assert len(cases) == 5
    # The fixture intentionally carries the previous 8192/512 preparation
    # caps; the runner evaluates it with the current 10240/1024 budget while
    # retaining the preparation values for provenance.
    assert runner.MAX_RESPONSE_TOKENS == 10240
    assert runner.TEACHER_PROPOSAL_MAX_NEW_TOKENS == 1024
    assert cases[0]["max_response_tokens"] == runner.MAX_RESPONSE_TOKENS
    assert cases[0]["teacher_replacement_max_new_tokens"] == runner.TEACHER_PROPOSAL_MAX_NEW_TOKENS
    assert cases[0]["prepared_dataset_max_response_tokens"] == 8192
    assert cases[0]["prepared_dataset_teacher_proposal_max_new_tokens"] == 512
    assert cases[2]["pre_step_response_token_ids"] == [0] * 0 + [0, 1, 2, 3, 4, 5, 6, 7]
    assert cases[2]["student_step_token_ids"] == [8, 9, 10, 11]
    assert cases[2]["keep_response_prefix_token_ids"] == list(range(12))
    assert cases[2]["pre_step_input_token_ids"] == [101, 102] + list(range(8))
    assert cases[2]["response_position_bin"] == "p3"
    assert cases[0]["source_response_token_ids"] is cases[1]["source_response_token_ids"]


def test_incorrect_source_is_supported_only_when_unambiguous(tmp_path: Path) -> None:
    _, cases = runner.load_dataset(_dataset(tmp_path, "incorrect"), "incorrect")
    assert all(case["source_outcome"] == "incorrect" for case in cases)


def test_sharding_is_disjoint_and_covers_all_cases() -> None:
    assert runner.worker_shards(313, 0, 8) == list(range(0, 313, 8))
    pieces = [set(runner.worker_shards(313, i, 8)) for i in range(8)]
    assert set.union(*pieces) == set(range(313))
    assert sum(map(len, pieces)) == 313


def test_paired_seeds_do_not_depend_on_worker_or_arm() -> None:
    assert runner.continuation_seed(123) == runner.continuation_seed(123)
    assert runner.proposal_seed(123, 1) != runner.proposal_seed(123, 2)
    assert runner.EXPANDED_ROLLOUTS_PER_GENERATE == 256
    assert runner.MAX_NUM_SEQS == 256


def test_gzip_arm_shard_resume_validation(tmp_path: Path) -> None:
    _, loaded = runner.load_dataset(_dataset(tmp_path), "correct")
    expected = loaded[:2]
    case = expected[0]
    prefix = list(case["keep_response_prefix_token_ids"])
    path = tmp_path / "part.json.gz"
    runner.atomic_gzip_json(
        path,
        {
            "config_signature": "sig",
            "arm": "replace",
            "phase": "continuations",
            "records": [{
                    "case_id": case["case_id"],
                    "case_index": case["case_index"],
                    "record_index": case["record_index"],
                    "source_response_text_sha256": case["source_response_text_sha256"],
                    "student_step_token_ids_sha256": case["student_step_token_ids_sha256"],
                    "response_prefix_token_ids": prefix,
                    "response_prefix_num_tokens": len(prefix),
                    "response_prefix_token_ids_sha256": runner.token_hash(prefix),
                    "input_token_ids_sha256": runner.token_hash(list(case["prompt_token_ids"]) + prefix),
                    "max_new_tokens": runner.MAX_RESPONSE_TOKENS - len(prefix),
                    "samples_per_arm": runner.SAMPLES_PER_ARM,
                    "temperature": runner.TEMPERATURE,
                    "top_p": runner.TOP_P,
                    "top_k": runner.TOP_K,
                    "request_seed": runner.continuation_seed(case["case_index"]),
                    "arm": "replace",
                    "outputs": [{
                        "sample_index": i,
                        "continuation_token_ids": [],
                        "continuation_num_tokens": 0,
                        "correct": 0,
                        "reward": 0,
                    } for i in range(128)],
                    "rewards": [0 for _ in range(128)],
                    "reward_sum": 0,
                }],
            "skipped": [{"case_id": expected[1]["case_id"],
                         "case_index": expected[1]["case_index"], "reason": "cap"}],
            "target_cases": 2,
            "generated_outputs": 128,
        },
    )
    assert runner._valid_arm_shard(path, expected, "sig", "replace")
    assert not runner._valid_arm_shard(path, expected, "other", "replace")


def test_max_model_len_covers_prompt_plus_full_response(tmp_path: Path) -> None:
    _, cases = runner.load_dataset(_dataset(tmp_path), "correct")
    assert runner.derive_max_model_len(cases) >= max(
        len(c["prompt_token_ids"]) + runner.MAX_RESPONSE_TOKENS for c in cases
    )


def test_continuation_record_keeps_all_128_rewards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, cases = runner.load_dataset(_dataset(tmp_path), "correct")
    case = cases[0]

    def fake_completion_result(*, sample_index, **_: object) -> dict[str, object]:
        correct = int(sample_index % 2 == 0)
        return {
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
            "correct": correct,
            "format_valid": 1,
            "extracted_answer": "1" if correct else "0",
            "truncated_by_response_cap": False,
        }

    monkeypatch.setattr(runner.base, "completion_result", fake_completion_result)
    output = SimpleNamespace(
        prompt_token_ids=case["keep_input_token_ids"],
        outputs=[SimpleNamespace(index=i) for i in range(runner.SAMPLES_PER_ARM)],
    )
    record = runner._continuation_record(
        case, "keep", None, output, object(), Path("student"), runner.continuation_seed(case["case_index"])
    )
    assert len(record["outputs"]) == 128
    assert record["reward_sum"] == 64
    assert record["accuracy_avg128"] == 0.5
    assert "continuation_token_ids" in record["outputs"][0]
    assert "full_response_text" in record["outputs"][0]
    assert "continuation_text" not in record["outputs"][0]
    assert "vllm_completion_text" not in record["outputs"][0]


def test_mock_worker_runs_proposals_and_all_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, cases = runner.load_dataset(_dataset(tmp_path), "correct")
    batches = runner.shard_plan(cases)
    root = tmp_path / "out"
    args = SimpleNamespace(
        outcome="correct", worker_id=0, world_size=1, gpu_id=0,
        validate_only=False, overwrite=False, min_free_gpu_gib=5.0,
        gpu_memory_utilization=0.5, heartbeat_seconds=30.0,
        memory_check_seconds=2.0, student_model=Path("student"),
        teacher_model=Path("teacher"), arm="all",
    )
    class _Tok:
        def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            del token_ids, skip_special_tokens, clean_up_tokenization_spaces
            return "A"

    monkeypatch.setattr(runner, "_load_tokenizer", lambda _: _Tok())
    monkeypatch.setattr(runner, "_new_llm", lambda *args: object())
    monkeypatch.setattr(runner, "_check_headroom", lambda *args: {})

    def fake_proposal(case: dict[str, object], *_: object) -> dict[str, object]:
        return {
            "case_id": case["case_id"], "case_index": case["case_index"],
            "record_index": case["record_index"],
            "replacement_step_token_ids": [65],
            "replacement_step_token_ids_sha256": runner.token_hash([65]),
            "replacement_step_decoded_text": "A", "teacher_model": "teacher",
            "request_seed": 1,
        }

    monkeypatch.setattr(runner, "_proposal_record", fake_proposal)
    monkeypatch.setattr(
        runner, "_generate",
        lambda _llm, prompts, _params, _args, _stage: [
            SimpleNamespace(prompt_token_ids=p, outputs=[SimpleNamespace(token_ids=[65], finish_reason="stop", stop_reason=None)])
            for p in prompts
        ],
    )
    proposal_map, failures = runner.run_proposals(
        args, root, cases, batches, "sig", runner.derive_max_model_len(cases)
    )
    assert len(proposal_map) == 5 and not failures

    monkeypatch.setattr(
        runner, "_completion_item",
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
    monkeypatch.setattr(
        runner, "_generate",
        lambda _llm, prompts, _params, _args, _stage: [
            SimpleNamespace(prompt_token_ids=p, outputs=[SimpleNamespace(index=i) for i in range(128)])
            for p in prompts
        ],
    )
    runner.run_continuations(
        args, root, cases, batches, "sig", runner.derive_max_model_len(cases),
        proposal_map, failures,
    )
    progress = json.loads((root / "progress" / "worker_00.json").read_text(encoding="utf-8"))
    assert progress["proposal"]["completed_cases"] == len(cases)
    assert progress["proposal"]["total_cases"] == len(cases)
    for arm in runner.ARMS:
        assert runner._valid_arm_shard(
            runner._arm_paths(root, 0, arm, 0), batches[0], "sig", arm
        )
