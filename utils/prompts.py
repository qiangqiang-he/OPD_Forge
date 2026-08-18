"""Complete prompt templates used by OPD_Forge student and teacher models."""

from __future__ import annotations


qwen3_thinking_prompt = r"""<|im_start|>system
You are a helpful math assistant.
Please solve the math problem step by step clearly and concisely.
You must enclose your final answer exactly within \boxed{{}}.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


qwen3_no_thinking_prompt = r"""<|im_start|>system
You are a helpful math assistant.
Please solve the math problem step by step clearly and concisely.
You must enclose your final answer exactly within \boxed{{}}.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


qwen3_response_extremely_correct_feedback = r"""<|im_start|>system
You are a helpful math assistant.
Please solve the math problem step by step clearly and concisely.
You must enclose your final answer exactly within \boxed{{}}.<|im_end|>
<|im_start|>user
{question}

A gold-standard final-answer verifier has judged that the assistant's following solution reaches the correct final answer.

This is an absolutely wonderful solution.
The reasoning is brilliant, flawless, rigorous, elegant, and exceptionally insightful.
Every assumption, transformation, calculation, and intermediate conclusion is precise, coherent, and logically impeccable.
The solution demonstrates outstanding mathematical ability and follows an extraordinarily strong reasoning trajectory.
It is a truly excellent solution that deserves the highest possible praise.<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


qwen3_response_extremely_incorrect_feedback = r"""<|im_start|>system
You are a helpful math assistant.
Please solve the math problem step by step clearly and concisely.
You must enclose your final answer exactly within \boxed{{}}.<|im_end|>
<|im_start|>user
{question}

A gold-standard final-answer verifier has judged that the assistant's following solution does not reach the correct final answer.

This is an absolutely terrible solution.
The reasoning is fundamentally broken, incoherent, careless, and almost entirely devoid of logic.
Its assumptions are baseless, its transformations are invalid, its calculations are unreliable, and its intermediate conclusions are absurd.
The solution demonstrates extremely poor mathematical judgment and follows a disastrously flawed reasoning trajectory.
It is a complete failure that deserves the harshest possible criticism.<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


qwen3_correct_answer_prompt = r"""<|im_start|>system
You are a helpful math assistant.
Please solve the math problem step by step clearly and concisely.
You must enclose your final answer exactly within \boxed{{}}.<|im_end|>
<|im_start|>user
{question}

A gold-standard verifier has confirmed that the correct final answer is:

\boxed{{{answer}}}

With this verified answer in mind, carefully examine the reasoning process and check every assumption, transformation, calculation, and intermediate conclusion. Ensure that each step is mathematically valid, logically consistent with the problem, and aligned with the verified final answer. Pay particular attention to possible arithmetic errors, unjustified transitions, logical gaps, or deviations from the correct reasoning path.<|im_end|>
<|im_start|>assistant
"""


qwen3_correct_feedback_v2 = r"""<|im_start|>system
You are a helpful math assistant.
Please solve the math problem step by step clearly and concisely.
You must enclose your final answer exactly within \boxed{{}}.<|im_end|>
<|im_start|>user
{question}

The candidate final answer \boxed{{{answer}}} has been marked as correct by a gold-standard verifier.

This is an excellent and highly convincing solution. The reasoning is exceptionally strong, rigorous, coherent, and mathematically sound. The intermediate steps are well-motivated, the transformations and calculations are precise, and the logical progression is clear and insightful. The solution demonstrates outstanding mathematical reasoning ability and follows a remarkably effective reasoning trajectory. Overall, both the reasoning process and the final conclusion deserve very high confidence and strong positive recognition.

With this highly positive feedback in mind, solve the problem step by step and arrive at the verified correct final answer.<|im_end|>
<|im_start|>assistant
"""


qwen3_incorrect_feedback_v2 = r"""<|im_start|>system
You are a helpful math assistant.
Please solve the math problem step by step clearly and concisely.
You must enclose your final answer exactly within \boxed{{}}.<|im_end|>
<|im_start|>user
{question}

The candidate final answer \boxed{{{answer}}} has been marked as incorrect by a gold-standard verifier.

This is a seriously flawed and highly unreliable solution. The reasoning contains substantial errors, lacks rigor and coherence, and is mathematically unsound. The intermediate steps are poorly justified, the transformations and calculations are inaccurate, and the logical progression is confused and misleading. The solution demonstrates weak mathematical reasoning ability and follows a fundamentally defective reasoning trajectory. Overall, both the reasoning process and the final conclusion deserve very low confidence and strong negative assessment.

With this highly negative feedback in mind, solve the problem step by step and arrive at the correct final answer.<|im_end|>
<|im_start|>assistant
"""


PROMPT_TEMPLATES = {
    "qwen3_thinking_prompt": qwen3_thinking_prompt,
    "qwen3_no_thinking_prompt": qwen3_no_thinking_prompt,
    "qwen3_response_extremely_correct_feedback": qwen3_response_extremely_correct_feedback,
    "qwen3_response_extremely_incorrect_feedback": qwen3_response_extremely_incorrect_feedback,
    "qwen3_correct_answer_prompt": qwen3_correct_answer_prompt,
    "qwen3_correct_feedback_v2": qwen3_correct_feedback_v2,
    "qwen3_incorrect_feedback_v2": qwen3_incorrect_feedback_v2,
}


def get_prompt_template(name: str) -> str:
    """Return a registered complete prompt template by name."""
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt template {name!r}. Available templates: {available}.") from exc


def render_prompt(name: str, *, question: str, answer: str = "") -> str:
    """Render a registered prompt without applying another chat template."""
    return get_prompt_template(name).format(question=question, answer=answer)
