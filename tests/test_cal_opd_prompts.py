"""Contracts for Cal-OPD's mode-matched privileged-feedback prompts."""

import pytest

from utils.prompts import (
    get_cal_privileged_feedback_prompt_names,
    render_prompt,
)


CAL_PROMPTS = (
    ("qwen3_positive_privileged_feedback_thinking", True),
    ("qwen3_negative_privileged_feedback_thinking", True),
    ("qwen3_positive_privileged_feedback_no_thinking", False),
    ("qwen3_negative_privileged_feedback_no_thinking", False),
)
THINKING_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
NO_THINKING_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_cal_opd_feedback_prompts_use_the_requested_thinking_mode():
    question = "What is 1+1?"
    for prompt_name, thinking in CAL_PROMPTS:
        rendered = render_prompt(prompt_name, question=question)
        expected_prefix = THINKING_ASSISTANT_PREFIX if thinking else NO_THINKING_ASSISTANT_PREFIX
        assert rendered.endswith(expected_prefix)
        assert ("<think>" not in rendered) is thinking
        assert rendered.count(question) == 1
        assert rendered.count("<|im_start|>system") == 1
        assert rendered.count("<|im_start|>user") == 1
        assert rendered.count("<|im_start|>assistant") == 1
        assert "{question}" not in rendered


@pytest.mark.parametrize(
    ("teacher_prompt", "expected"),
    (
        (
            "qwen3_thinking_prompt",
            (
                "qwen3_positive_privileged_feedback_thinking",
                "qwen3_negative_privileged_feedback_thinking",
            ),
        ),
        (
            "qwen3_no_thinking_prompt",
            (
                "qwen3_positive_privileged_feedback_no_thinking",
                "qwen3_negative_privileged_feedback_no_thinking",
            ),
        ),
    ),
)
def test_cal_opd_selects_privileged_prompts_from_teacher_mode(teacher_prompt, expected):
    assert get_cal_privileged_feedback_prompt_names(teacher_prompt) == expected


def test_cal_opd_rejects_unknown_teacher_prompt_mode():
    with pytest.raises(ValueError, match="supported thinking-mode teacher prompt"):
        get_cal_privileged_feedback_prompt_names("custom_teacher_prompt")
