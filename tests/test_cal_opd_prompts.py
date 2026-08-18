"""Contracts for Cal-OPD's counterfactual no-thinking prompts."""

from utils.prompts import render_prompt


CAL_PROMPTS = (
    "qwen3_response_extremely_correct_feedback",
    "qwen3_response_extremely_incorrect_feedback",
)
NO_THINKING_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_cal_opd_feedback_prompts_use_the_no_thinking_prefix():
    question = "What is 1+1?"
    for prompt_name in CAL_PROMPTS:
        rendered = render_prompt(prompt_name, question=question)
        assert rendered.endswith(NO_THINKING_ASSISTANT_PREFIX)
        assert rendered.count(question) == 1
        assert rendered.count("<|im_start|>system") == 1
        assert rendered.count("<|im_start|>user") == 1
        assert rendered.count("<|im_start|>assistant") == 1
        assert "{question}" not in rendered
