"""Binary math reward adapter for OPD_Forge."""

from __future__ import annotations

from typing import Any

from utils.math_verifier import extract_final_answer, verify_response_answer


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return the binary math reward plus parsing diagnostics."""

    del data_source, extra_info
    _, valid_format = extract_final_answer(solution_str)
    score = verify_response_answer(solution_str, str(ground_truth))
    return {
        "score": float(score),
        "acc": float(score),
        "format_valid": float(valid_format),
    }
