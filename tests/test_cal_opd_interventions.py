"""Data-construction tests for answer and solution Cal-OPD ablations."""

import asyncio
import json
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from utils.cal_opd_interventions import (
    choose_cal_opd_wrong_solution_indices,
    make_cal_opd_wrong_answer,
)


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        padding,
        truncation,
    ):
        assert add_special_tokens is False
        assert padding is False
        assert truncation is False
        return {"input_ids": [self.encode(text) for text in texts]}

    def decode(self, token_ids):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _dataset_config(intervention_type):
    return OmegaConf.create(
        {
            "student_prompt": "qwen3_thinking_prompt",
            "teacher_prompt": "qwen3_thinking_prompt",
            "cal_intervention_type": intervention_type,
            "prompt_key": "prompt",
            "filter_overlong_prompts": False,
            "shuffle": False,
            "seed": 42,
        }
    )


def test_wrong_answer_is_seeded_random_and_preserves_latex_digit_structure():
    answers = ["34", "0", "-8007", r"\frac{123}{405}"]
    for index, answer in enumerate(answers):
        first = make_cal_opd_wrong_answer(answer, seed=42, record_index=index)
        second = make_cal_opd_wrong_answer(answer, seed=42, record_index=index)
        assert first == second
        assert len(first) == len(answer)
        before_digits = [character for character in answer if character.isdigit()]
        after_digits = [character for character in first if character.isdigit()]
        assert len(before_digits) == len(after_digits)
        assert all(
            before != after
            for before, after in zip(before_digits, after_digits, strict=True)
        )


def test_wrong_answer_without_digits_has_a_deterministic_changed_fallback():
    assert make_cal_opd_wrong_answer("x", seed=42, record_index=0) == (
        r"\left(x\right)+1"
    )


def test_wrong_solution_is_nearest_length_but_excludes_same_question_and_answer():
    records = [
        {"question": "q0", "answer": "a", "solution": "s0"},
        {"question": "q1", "answer": "a", "solution": "s1"},
        {"question": "q2", "answer": "b", "solution": "s2"},
        {"question": "q3", "answer": "c", "solution": "s3"},
    ]
    assert choose_cal_opd_wrong_solution_indices(records, [100, 101, 107, 130]) == [
        2,
        2,
        1,
        2,
    ]


def test_empty_validation_solutions_do_not_require_false_privileges():
    records = [
        {"question": "q0", "answer": "a", "solution": ""},
        {"question": "q1", "answer": "b", "solution": ""},
    ]
    assert choose_cal_opd_wrong_solution_indices(records, [0, 0]) == [-1, -1]


def test_custom_dataset_builds_reproducible_answer_and_solution_privileges(tmp_path):
    from utils.custom_dataset import CustomDataset

    records = [
        {"question": "q0", "answer": "34", "solution": "a" * 20},
        {"question": "q1", "answer": "56", "solution": "b" * 24},
        {"question": "q2", "answer": "78", "solution": "c" * 80},
    ]
    data_path = tmp_path / "train.json"
    data_path.write_text(json.dumps(records), encoding="utf-8")
    tokenizer = _CharacterTokenizer()

    answer_data = CustomDataset(
        str(data_path),
        tokenizer,
        _dataset_config("answer"),
        max_samples=-1,
    )
    answer_row = answer_data.dataframe[0]
    assert answer_row["cal_ground_truth_answer"] == "34"
    assert answer_row["cal_wrong_answer"] != "34"
    assert len(answer_row["cal_wrong_answer"]) == 2

    solution_data = CustomDataset(
        str(data_path),
        tokenizer,
        _dataset_config("solution"),
        max_samples=-1,
    )
    solution_row = solution_data.dataframe[0]
    assert solution_row["cal_privileged_solution"] == records[0]["solution"]
    assert solution_row["cal_wrong_solution"] == records[1]["solution"]


class _FakeTeacherManager:
    def __init__(self, response_ids, diagnostic_topk):
        self.response_ids = list(response_ids)
        self.diagnostic_topk = int(diagnostic_topk)
        self.calls = []

    def get_teacher_prompt_length(self, routing_key=None):
        del routing_key
        return 8192

    async def compute_teacher_logprobs_single(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["sequence_ids"][-len(self.response_ids) :] == self.response_ids
        prompt_length = int(kwargs["student_prompt_length"])
        response_length = int(kwargs["response_length"])
        sequence_length = prompt_length + response_length
        causal_start = prompt_length - 1
        sampled_ids = torch.zeros((sequence_length, 1), dtype=torch.int32)
        sampled_ids[causal_start : causal_start + response_length, 0] = torch.tensor(
            self.response_ids,
            dtype=torch.int32,
        )
        sampled_logprobs = torch.full(
            (sequence_length, 1),
            -1.0,
            dtype=torch.float32,
        )
        topk_ids = sampled_ids.expand(-1, self.diagnostic_topk).clone()
        topk_logprobs = torch.full(
            (sequence_length, self.diagnostic_topk),
            -2.0,
            dtype=torch.float32,
        )
        return (
            sampled_ids,
            sampled_logprobs,
            topk_ids,
            topk_logprobs,
            sampled_logprobs.clone(),
            None,
            {"teacher_engine_s": 0.0, "teacher_logprob_extract_s": 0.0},
        )


class _FakeStudentClient:
    def __init__(self, response_ids, diagnostic_topk):
        self.response_ids = list(response_ids)
        self.diagnostic_topk = int(diagnostic_topk)

    async def generate(self, *, prompt_ids, **kwargs):
        del kwargs
        sequence_length = len(prompt_ids)
        response_length = len(self.response_ids)
        causal_start = sequence_length - response_length - 1
        ids = torch.zeros(
            (sequence_length, self.diagnostic_topk),
            dtype=torch.int32,
        )
        ids[causal_start : causal_start + response_length, :] = torch.tensor(
            self.response_ids,
            dtype=torch.int32,
        ).unsqueeze(-1)
        logprobs = torch.full(
            (sequence_length, self.diagnostic_topk),
            -2.0,
            dtype=torch.float32,
        )
        return SimpleNamespace(
            extra_fields={
                "prompt_ids": ids.tolist(),
                "prompt_logprobs": logprobs.tolist(),
            }
        )


@pytest.mark.parametrize(
    ("intervention_type", "positive_marker", "negative_marker"),
    (
        ("inst", "carefully and thoroughly", "quickly and directly"),
        ("eval", "reaches the correct final answer", "does not reach"),
        ("answer", "CORRECT-34", "WRONG-89"),
        ("solution", "CORRECT-SOLUTION", "UNRELATED-SOLUTION"),
    ),
)
def test_agent_loop_scores_exact_response_ids_under_each_intervention_pair(
    intervention_type,
    positive_marker,
    negative_marker,
):
    from utils.prompts import render_prompt
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
        AgentLoopWorker,
    )

    tokenizer = _CharacterTokenizer()
    question = "Compute 17 + 17."
    prompt_text = render_prompt("qwen3_thinking_prompt", question=question)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = [7001, 7002, 7003]
    diagnostic_topk = 2
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.teacher_key = "data_source"
    worker.tokenizer = tokenizer
    worker.config = OmegaConf.create(
        {
            "algorithm": {"name": "cal_opd"},
            "teacher_prompt": "qwen3_thinking_prompt",
            "cal_intervention_type": intervention_type,
            "distillation": {
                "distillation_loss": {"diagnostic_topk": diagnostic_topk}
            },
        }
    )
    manager = _FakeTeacherManager(response_ids, diagnostic_topk)
    worker.teacher_server_manager = manager
    worker.llm_client = _FakeStudentClient(response_ids, diagnostic_topk)

    async def tokenize_preformatted_prompt(text):
        return tokenizer.encode(text, add_special_tokens=False)

    worker.tokenize_preformatted_prompt = tokenize_preformatted_prompt
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        metrics=AgentLoopMetrics(),
    )
    asyncio.run(
        worker._compute_teacher_logprobs(
            output,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            validate=False,
            sample_kwargs={
                "data_source": "math",
                "teacher_prompt_text": prompt_text,
                "cal_question": question,
                "cal_ground_truth_answer": "CORRECT-34",
                "cal_wrong_answer": "WRONG-89",
                "cal_privileged_solution": "CORRECT-SOLUTION",
                "cal_wrong_solution": "UNRELATED-SOLUTION",
            },
        )
    )

    assert len(manager.calls) == 3
    base_call, positive_call, negative_call = manager.calls
    assert base_call["sequence_ids"][:-3] == prompt_ids
    positive_prompt = tokenizer.decode(positive_call["sequence_ids"][:-3])
    negative_prompt = tokenizer.decode(negative_call["sequence_ids"][:-3])
    assert positive_marker in positive_prompt
    assert negative_marker in negative_prompt
    for call in manager.calls:
        assert call["sequence_ids"][-3:] == response_ids
        assert call["student_prompt_length"] == len(prompt_ids)
        assert call["response_length"] == len(response_ids)
    assert "cal_positive_teacher_logprobs" in output.extra_fields
    assert "cal_negative_teacher_logprobs" in output.extra_fields
