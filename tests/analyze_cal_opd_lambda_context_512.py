"""Analyze matched Cal-OPD lambda 5/10/20 rollouts with token context."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from verl.trainer.distillation.losses import compute_calibrated_opd_advantage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUMMARY_DIR = OUTPUT_ROOT / "cal_opd_lambda_context_512_summary"
TOKENIZER_PATH = PROJECT_ROOT.parent / "FrugalRL/models/Qwen3-1.7B"
MODES = ("no_thinking", "thinking")
LAMBDAS = (5, 10, 20)
LOW_PROB_THRESHOLDS = (0.1, 0.01, 0.001)
LARGE_ADVANTAGE_THRESHOLDS = (1.0, 2.0, 4.0)
SYMBOLS = ("1", "2", "-", "=")


def _run_dir(mode: str) -> Path:
    return OUTPUT_ROOT / f"temp_cal_opd_qwen3_4b_to_1p7b_{mode}_lambda_coverage_512"


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _count(selection: torch.Tensor) -> int:
    return int(selection.sum().item())


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float()
    if values.numel() == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
    quantiles = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
    }


def _decode_single_token_ids(tokenizer, token_ids: list[int]) -> dict[int, str]:
    decoded: dict[int, str] = {}
    for start in range(0, len(token_ids), 4096):
        chunk = token_ids[start : start + 4096]
        pieces = tokenizer.batch_decode(
            [[token_id] for token_id in chunk],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decoded.update(zip(chunk, (str(piece) for piece in pieces), strict=True))
    return decoded


def _nearest_nonspace(text: str, *, from_right: bool) -> str:
    iterable = reversed(text) if from_right else iter(text)
    return next((character for character in iterable if not character.isspace()), "")


def _inside_math_context(left: str) -> bool:
    """Heuristically identify an unmatched inline or block math opener."""

    return (
        left.count("$") % 2 == 1
        or left.count("$$") % 2 == 1
        or left.count("\\[") > left.count("\\]")
        or left.count("\\(") > left.count("\\)")
    )


def _classify_digit(symbol: str, left: str, right: str) -> str:
    line_prefix = left.rsplit("\n", 1)[-1]
    if re.fullmatch(r"\s*(?:[-*+]\s*)?", line_prefix) and re.match(
        r"\s*[.)、:]\s+", right
    ):
        return "list_marker"
    if (left and left[-1].isdigit()) or (right and right[0].isdigit()):
        return "numeric_sequence"
    if re.search(r"(?:step|case|option|part|point|item|no\.?|number)\s*$", left, re.I):
        return "step_or_item_reference"
    if re.match(r"(?:st|nd|rd|th)\b", right, re.I):
        return "ordinal_reference"
    math_window = left[-24:] + symbol + right[:24]
    if (
        re.search(r"[=+\-*/^<>×÷±√]", math_window)
        or re.search(r"\\(?:frac|sqrt|cdot|times|equiv|leq|geq)", math_window)
        or left.count("$") % 2 == 1
    ):
        return "math_expression"
    return "standalone_number"


def _classify_minus(left: str, right: str) -> str:
    line_prefix = left.rsplit("\n", 1)[-1]
    if re.fullmatch(r"\s*", line_prefix) and re.match(r"\s+\S", right):
        return "markdown_list_marker"
    if (left and left[-1] == "-") or (right and right[0] == "-"):
        return "repeated_dash_or_separator"
    immediate_left = left[-1:] or ""
    immediate_right = right[:1] or ""
    if immediate_left.isalpha() and immediate_right.isalpha():
        return "word_hyphen"
    left_operand = _nearest_nonspace(left, from_right=True)
    right_operand = _nearest_nonspace(right, from_right=False)
    operand_chars = set(")]}|!")
    right_operand_chars = set("([{")
    left_is_operand = left_operand.isalnum() or left_operand in operand_chars
    right_is_operand = (
        right_operand.isalnum()
        or right_operand in right_operand_chars
        or right_operand in {"\\", "√"}
    )
    if _inside_math_context(left):
        if right_is_operand and (not left_operand or left_operand in "=+-*/^<>,([{$"):
            return "unary_negative"
        return "subtraction_operator"
    if left_is_operand and right_is_operand:
        if left_operand.isdigit() and right_operand.isdigit() and not re.search(
            r"[$=+*/^\\]", left[-20:] + right[:20]
        ):
            return "numeric_range_or_subtraction"
        return "subtraction_operator"
    if right_is_operand and (not left_operand or left_operand in "=+-*/^<>,([{$"):
        return "unary_negative"
    return "punctuation_dash_or_other"


def _classify_equals(left: str, right: str) -> str:
    if (left and left[-1] in "=<>!") or (right and right[0] in "=>"):
        return "comparison_assignment_or_arrow"
    line = left.rsplit("\n", 1)[-1] + "=" + right.split("\n", 1)[0]
    if line.count("=") >= 3 and re.fullmatch(r"\s*=+\s*", line):
        return "decorative_separator"
    if _inside_math_context(left):
        return "formula_equality_or_assignment"
    # Ignore surrounding Markdown/LaTeX delimiters when identifying the two
    # sides of an equality (for example ``**Area** = 20`` or ``$|3x| = 12$``).
    left_operand = _nearest_nonspace(left.rstrip(" `*$"), from_right=True)
    right_operand = _nearest_nonspace(right.lstrip(" `*$"), from_right=False)
    left_is_operand = (
        left_operand.isalnum()
        or left_operand in ")]}'\"|"
        or left_operand == "\\"
    )
    right_is_operand = (
        right_operand.isalnum()
        or right_operand in "([{\"'|"
        or right_operand in {"\\", "√", "+", "-", "±"}
    )
    if left_is_operand and right_is_operand:
        return "formula_equality_or_assignment"
    # A standalone equals sign in these natural-language math rollouts still
    # denotes equality even when Markdown delimiters obscure one operand.
    return "formula_equality_or_assignment"


def _classify_symbol(symbol: str, left: str, right: str) -> str:
    if symbol in {"1", "2"}:
        return _classify_digit(symbol, left, right)
    if symbol == "-":
        return _classify_minus(left, right)
    return _classify_equals(left, right)


def _context_for_position(
    *, tokenizer, ids: list[int], column: int, symbol: str, window: int = 120
) -> tuple[str, str, str]:
    # Decoding each side separately preserves the exact local token boundary,
    # including whitespace carried by the selected token piece.
    left_piece = tokenizer.decode(
        ids[max(0, column - 48) : column],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    current_piece = tokenizer.decode(
        [ids[column]],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    right_piece = tokenizer.decode(
        ids[column + 1 : column + 49],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    offset = current_piece.find(symbol)
    if offset < 0:
        offset = max(0, len(current_piece) - len(symbol))
    left = (left_piece + current_piece[:offset])[-window:]
    right = (current_piece[offset + len(symbol) :] + right_piece)[:window]
    context = (left + f"⟦{symbol}⟧" + right).replace("\n", "\\n")
    return left, right, context


def _top_token_changes(
    *,
    response_token_ids: torch.Tensor,
    valid: torch.Tensor,
    changes: torch.Tensor,
    decoded_by_id: dict[int, str],
    top_k: int = 40,
) -> list[dict[str, Any]]:
    selected_ids = response_token_ids[valid].long()
    selected_changes = changes[valid].float()
    nonzero = selected_changes.gt(0)
    selected_ids = selected_ids[nonzero]
    selected_changes = selected_changes[nonzero]
    unique_ids, inverse = torch.unique(selected_ids, sorted=True, return_inverse=True)
    totals = torch.zeros(unique_ids.numel(), dtype=torch.float64)
    totals.scatter_add_(0, inverse, selected_changes.double())
    counts = torch.zeros(unique_ids.numel(), dtype=torch.int64)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
    merged: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_change": 0.0, "count": 0, "token_ids": []}
    )
    for index, token_id in enumerate(unique_ids.tolist()):
        token = decoded_by_id[int(token_id)].strip()
        if not token:
            continue
        row = merged[token]
        row["total_change"] += float(totals[index].item())
        row["count"] += int(counts[index].item())
        row["token_ids"].append(int(token_id))
    rows = [
        {
            "token": token,
            "total_l1_removed": values["total_change"],
            "changed_occurrence_count": values["count"],
            "mean_l1_removed": _safe_ratio(values["total_change"], values["count"]),
            "token_ids": values["token_ids"],
        }
        for token, values in merged.items()
    ]
    rows.sort(key=lambda row: (-row["total_l1_removed"], -row["changed_occurrence_count"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[:top_k]


def _double_low_analysis(
    *,
    valid_nonzero: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    base_advantage: torch.Tensor,
    calibrated: dict[int, torch.Tensor],
) -> dict[str, Any]:
    base_abs = base_advantage.abs()
    rows = []
    for low_probability in LOW_PROB_THRESHOLDS:
        double_low = (
            valid_nonzero
            & student_log_probs.lt(math.log(low_probability))
            & teacher_log_probs.lt(math.log(low_probability))
        )
        for minimum_abs_advantage in LARGE_ADVANTAGE_THRESHOLDS:
            selection = double_low & base_abs.ge(minimum_abs_advantage)
            count = _count(selection)
            base_l1 = float(base_abs[selection].sum().item())
            row: dict[str, Any] = {
                "max_probability": low_probability,
                "min_abs_advantage": minimum_abs_advantage,
                "count": count,
                "fraction_of_base_nonzero": _safe_ratio(count, _count(valid_nonzero)),
                "positive_count": _count(selection & base_advantage.gt(0)),
                "negative_count": _count(selection & base_advantage.lt(0)),
                "base_l1": base_l1,
                "abs_advantage_distribution": _distribution(base_abs[selection]),
                "per_lambda": {},
            }
            for cal_lambda, advantage in calibrated.items():
                cal_abs = advantage.abs()
                changed = selection & cal_abs.lt(base_abs)
                zeroed = selection & advantage.eq(0)
                retained_l1 = float(cal_abs[selection].sum().item())
                row["per_lambda"][str(cal_lambda)] = {
                    "changed_count": _count(changed),
                    "changed_ratio": _safe_ratio(_count(changed), count),
                    "zeroed_count": _count(zeroed),
                    "zeroed_ratio": _safe_ratio(_count(zeroed), count),
                    "retained_l1_ratio": _safe_ratio(retained_l1, base_l1),
                    "removed_l1_ratio": 1.0 - _safe_ratio(retained_l1, base_l1),
                }
            rows.append(row)
    return {
        "definition": "student probability and teacher probability are both below the threshold",
        "rows": rows,
    }


def _magnitude_analysis(
    *,
    valid_nonzero: torch.Tensor,
    base_advantage: torch.Tensor,
    calibrated: dict[int, torch.Tensor],
) -> dict[str, Any]:
    base_abs = base_advantage.abs()
    bins = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, math.inf))
    output: dict[str, Any] = {}
    for cal_lambda, advantage in calibrated.items():
        zeroed = valid_nonzero & advantage.eq(0)
        surviving = valid_nonzero & advantage.ne(0)
        bin_rows = []
        for lower, upper in bins:
            selection = valid_nonzero & base_abs.ge(lower)
            if math.isfinite(upper):
                selection &= base_abs.lt(upper)
            bin_count = _count(selection)
            bin_base_l1 = float(base_abs[selection].sum().item())
            bin_retained_l1 = float(advantage[selection].abs().sum().item())
            bin_rows.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": bin_count,
                    "zeroed_count": _count(selection & zeroed),
                    "zeroed_ratio": _safe_ratio(_count(selection & zeroed), bin_count),
                    "retained_l1_ratio": _safe_ratio(bin_retained_l1, bin_base_l1),
                }
            )
        output[str(cal_lambda)] = {
            "zeroed_abs_advantage": _distribution(base_abs[zeroed]),
            "surviving_abs_advantage": _distribution(base_abs[surviving]),
            "bins": bin_rows,
        }
    return output


def _symbol_context_analysis(
    *,
    tokenizer,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    valid_nonzero: torch.Tensor,
    base_advantage: torch.Tensor,
    teacher_self_deviation: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    calibrated: dict[int, torch.Tensor],
    decoded_by_id: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    symbol_ids: dict[str, set[int]] = {symbol: set() for symbol in SYMBOLS}
    for token_id, decoded in decoded_by_id.items():
        normalized = decoded.strip()
        if normalized in symbol_ids:
            symbol_ids[normalized].add(token_id)

    occurrences: list[dict[str, Any]] = []
    for row_index in range(response_token_ids.shape[0]):
        valid_columns = torch.nonzero(response_mask[row_index], as_tuple=False).flatten()
        if valid_columns.numel() == 0:
            continue
        last_column = int(valid_columns[-1].item())
        ids = response_token_ids[row_index, : last_column + 1].tolist()
        for column in valid_columns.tolist():
            if not valid_nonzero[row_index, column]:
                continue
            token_id = int(response_token_ids[row_index, column].item())
            symbol = next(
                (candidate for candidate, ids_set in symbol_ids.items() if token_id in ids_set),
                None,
            )
            if symbol is None:
                continue
            left, right, context = _context_for_position(
                tokenizer=tokenizer, ids=ids, column=int(column), symbol=symbol
            )
            occurrences.append(
                {
                    "row": row_index,
                    "column": int(column),
                    "symbol": symbol,
                    "category": _classify_symbol(symbol, left, right),
                    "context": context,
                }
            )

    details: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []
    base_abs = base_advantage.abs()
    for symbol in SYMBOLS:
        symbol_occurrences = [row for row in occurrences if row["symbol"] == symbol]
        symbol_detail: dict[str, Any] = {
            "token_ids": sorted(symbol_ids[symbol]),
            "base_nonzero_occurrence_count": len(symbol_occurrences),
            "categories": {},
            "per_lambda": {},
            "transitions": {},
        }
        categories = sorted({str(row["category"]) for row in symbol_occurrences})
        for category in categories:
            category_rows = [row for row in symbol_occurrences if row["category"] == category]
            symbol_detail["categories"][category] = {"base_nonzero_count": len(category_rows)}

        previous_zero: set[tuple[int, int]] = set()
        for cal_lambda in LAMBDAS:
            advantage = calibrated[cal_lambda]
            zero_rows = [
                row
                for row in symbol_occurrences
                if advantage[row["row"], row["column"]].item() == 0
            ]
            zero_keys = {(row["row"], row["column"]) for row in zero_rows}
            removed_l1 = sum(
                float(
                    base_abs[row["row"], row["column"]].item()
                    - advantage[row["row"], row["column"]].abs().item()
                )
                for row in symbol_occurrences
            )
            category_counts = defaultdict(int)
            for row in zero_rows:
                category_counts[str(row["category"])] += 1
            symbol_detail["per_lambda"][str(cal_lambda)] = {
                "zeroed_count": len(zero_rows),
                "zeroed_ratio": _safe_ratio(len(zero_rows), len(symbol_occurrences)),
                "total_l1_removed": removed_l1,
                "zeroed_category_counts": dict(sorted(category_counts.items())),
            }
            incremental = [
                row for row in zero_rows if (row["row"], row["column"]) not in previous_zero
            ]
            incremental_counts = defaultdict(int)
            for row in incremental:
                incremental_counts[str(row["category"])] += 1
            transition = "base_to_lambda5" if cal_lambda == 5 else f"lambda{cal_lambda // 2}_to_lambda{cal_lambda}"
            symbol_detail["transitions"][transition] = {
                "count": len(incremental),
                "category_counts": dict(sorted(incremental_counts.items())),
            }
            previous_zero = zero_keys

        lambda20_zero = [
            row
            for row in symbol_occurrences
            if calibrated[20][row["row"], row["column"]].item() == 0
        ]
        for category in categories:
            candidates = [row for row in lambda20_zero if row["category"] == category]
            candidates.sort(
                key=lambda row: -float(base_abs[row["row"], row["column"]].item())
            )
            for row in candidates[:3]:
                row_index, column = row["row"], row["column"]
                examples.append(
                    {
                        "type": "symbol",
                        "symbol": symbol,
                        "category": category,
                        "context": row["context"],
                        "base_advantage": float(base_advantage[row_index, column].item()),
                        "teacher_self_deviation": float(
                            teacher_self_deviation[row_index, column].item()
                        ),
                        "student_probability": float(student_log_probs[row_index, column].exp().item()),
                        "teacher_probability": float(teacher_log_probs[row_index, column].exp().item()),
                        "zeroed_at": [
                            cal_lambda
                            for cal_lambda in LAMBDAS
                            if calibrated[cal_lambda][row_index, column].item() == 0
                        ],
                    }
                )
        details[symbol] = symbol_detail
    return details, examples


def _double_low_examples(
    *,
    tokenizer,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    base_advantage: torch.Tensor,
    teacher_self_deviation: torch.Tensor,
    calibrated: dict[int, torch.Tensor],
    decoded_by_id: dict[int, str],
    limit: int = 24,
) -> list[dict[str, Any]]:
    selection = (
        response_mask
        & base_advantage.ne(0)
        & student_log_probs.lt(math.log(0.01))
        & teacher_log_probs.lt(math.log(0.01))
        & base_advantage.abs().ge(2.0)
    )
    positions = torch.nonzero(selection, as_tuple=False)
    if positions.numel() == 0:
        return []
    values = base_advantage.abs()[selection]
    top_indices = torch.topk(values, k=min(limit, values.numel())).indices
    examples = []
    for position_index in top_indices.tolist():
        row_index, column = positions[position_index].tolist()
        valid_columns = torch.nonzero(response_mask[row_index], as_tuple=False).flatten()
        ids = response_token_ids[row_index, : int(valid_columns[-1].item()) + 1].tolist()
        token_id = int(response_token_ids[row_index, column].item())
        token = decoded_by_id[token_id]
        display_symbol = token.strip() or token
        _, _, context = _context_for_position(
            tokenizer=tokenizer,
            ids=ids,
            column=column,
            symbol=display_symbol,
        )
        examples.append(
            {
                "type": "double_low_large_advantage",
                "token": token,
                "context": context,
                "base_advantage": float(base_advantage[row_index, column].item()),
                "abs_base_advantage": float(base_advantage[row_index, column].abs().item()),
                "teacher_self_deviation": float(teacher_self_deviation[row_index, column].item()),
                "student_probability": float(student_log_probs[row_index, column].exp().item()),
                "teacher_probability": float(teacher_log_probs[row_index, column].exp().item()),
                "calibrated_advantage": {
                    str(cal_lambda): float(calibrated[cal_lambda][row_index, column].item())
                    for cal_lambda in LAMBDAS
                },
                "zeroed_at": [
                    cal_lambda
                    for cal_lambda in LAMBDAS
                    if calibrated[cal_lambda][row_index, column].item() == 0
                ],
            }
        )
    return examples


def _analyze_mode(mode: str, tokenizer) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = _run_dir(mode)
    record = json.loads((run_dir / "cal_opd_lambda_stats.jsonl").read_text())
    tensors = torch.load(run_dir / "cal_opd_token_tensors.pt", map_location="cpu", weights_only=True)
    if int(tensors["sample_count"].item()) != 512:
        raise RuntimeError(f"Expected 512 samples for {mode}")

    response_token_ids = tensors["response_token_ids"].long()
    response_mask = tensors["response_mask"].bool()
    student_log_probs = tensors["student_log_probs"].float()
    teacher_log_probs = tensors["teacher_log_probs"].float()
    positive_teacher_log_probs = tensors["positive_teacher_log_probs"].float()
    negative_teacher_log_probs = tensors["negative_teacher_log_probs"].float()
    base_advantage = teacher_log_probs - student_log_probs
    valid_nonzero = response_mask & base_advantage.ne(0)
    calibrated = {
        cal_lambda: compute_calibrated_opd_advantage(
            student_log_probs,
            teacher_log_probs,
            positive_teacher_log_probs,
            negative_teacher_log_probs,
            cal_lambda=cal_lambda,
        )[0]
        for cal_lambda in LAMBDAS
    }
    teacher_self_deviation = compute_calibrated_opd_advantage(
        student_log_probs,
        teacher_log_probs,
        positive_teacher_log_probs,
        negative_teacher_log_probs,
    )[1]

    unique_ids = sorted(set(response_token_ids[response_mask].tolist()))
    decoded_by_id = _decode_single_token_ids(tokenizer, unique_ids)
    token_rankings = {
        str(cal_lambda): _top_token_changes(
            response_token_ids=response_token_ids,
            valid=response_mask,
            changes=base_advantage.abs() - calibrated[cal_lambda].abs(),
            decoded_by_id=decoded_by_id,
        )
        for cal_lambda in LAMBDAS
    }
    symbol_context, symbol_examples = _symbol_context_analysis(
        tokenizer=tokenizer,
        response_token_ids=response_token_ids,
        response_mask=response_mask,
        valid_nonzero=valid_nonzero,
        base_advantage=base_advantage,
        teacher_self_deviation=teacher_self_deviation,
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        calibrated=calibrated,
        decoded_by_id=decoded_by_id,
    )
    examples = symbol_examples + _double_low_examples(
        tokenizer=tokenizer,
        response_token_ids=response_token_ids,
        response_mask=response_mask,
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        base_advantage=base_advantage,
        teacher_self_deviation=teacher_self_deviation,
        calibrated=calibrated,
        decoded_by_id=decoded_by_id,
    )
    return (
        {
            "sample_count": int(record["sample_count"]),
            "valid_token_count": int(record["valid_token_count"]),
            "base_nonzero_token_count": int(record["base_nonzero_token_count"]),
            "mean_valid_tokens_per_response": float(record["mean_valid_tokens_per_response"]),
            "per_lambda": record["per_lambda"],
            "transitions": record["transitions"],
            "double_low_large_advantage": _double_low_analysis(
                valid_nonzero=valid_nonzero,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                base_advantage=base_advantage,
                calibrated=calibrated,
            ),
            "magnitude_profile": _magnitude_analysis(
                valid_nonzero=valid_nonzero,
                base_advantage=base_advantage,
                calibrated=calibrated,
            ),
            "top_token_l1_changes": token_rankings,
            "symbol_context": symbol_context,
        },
        examples,
    )


def _write_coverage_csv(summary: dict[str, Any]) -> None:
    rows = []
    for mode, data in summary["modes"].items():
        for cal_lambda in LAMBDAS:
            rows.append(
                {
                    "mode": mode,
                    "lambda": cal_lambda,
                    "sample_count": data["sample_count"],
                    "valid_token_count": data["valid_token_count"],
                    "base_nonzero_token_count": data["base_nonzero_token_count"],
                    **data["per_lambda"][str(cal_lambda)],
                }
            )
    with (SUMMARY_DIR / "lambda_coverage.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Cal-OPD lambda 5/10/20 context analysis (512 questions × 1 rollout)",
        "",
        "All retained-magnitude values use L1: `sum(abs(A_cal)) / sum(abs(A_OPD))`.",
        "",
    ]
    for mode, data in summary["modes"].items():
        lines.extend(
            [
                f"## {mode}",
                "",
                f"Samples: {data['sample_count']}; valid tokens: {data['valid_token_count']:,}; "
                f"mean tokens/response: {data['mean_valid_tokens_per_response']:.2f}; "
                f"base-nonzero tokens: {data['base_nonzero_token_count']:,}.",
                "",
                "| Lambda | Newly zero | Zero/response | Zero/base-nonzero | Retained L1 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for cal_lambda in LAMBDAS:
            row = data["per_lambda"][str(cal_lambda)]
            lines.append(
                f"| {cal_lambda} | {row['newly_zero_from_base_count']:,} | "
                f"{row['mean_newly_zero_tokens_per_response']:.2f} | "
                f"{row['newly_zero_from_base_ratio']:.2%} | "
                f"{row['retained_l1_magnitude_ratio']:.2%} |"
            )

        lines.extend(
            [
                "",
                "| Incremental transition | Additional zero tokens | Additional zero/response |",
                "|---|---:|---:|",
            ]
        )
        for transition in ("base_to_lambda5", "lambda5_to_lambda10", "lambda10_to_lambda20"):
            row = data["transitions"][transition]
            lines.append(
                f"| {transition} | {row['count']:,} | {row['mean_count_per_response']:.2f} |"
            )

        lines.extend(
            [
                "",
                "### Double-low, large-|A| sensitivity",
                "",
                "Both student and teacher probabilities are below `p max`; `changed` includes partial shrinkage, while `zeroed` means exactly zero.",
                "",
                "| p max | min abs A | subset count | Lambda | changed | zeroed | retained L1 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in data["double_low_large_advantage"]["rows"]:
            for cal_lambda in LAMBDAS:
                metric = row["per_lambda"][str(cal_lambda)]
                lines.append(
                    f"| {row['max_probability']:.3g} | {row['min_abs_advantage']:.0f} | "
                    f"{row['count']:,} | {cal_lambda} | {metric['changed_ratio']:.2%} | "
                    f"{metric['zeroed_ratio']:.2%} | {metric['retained_l1_ratio']:.2%} |"
                )

        lines.extend(
            [
                "",
                "### Coverage by original absolute-advantage magnitude",
                "",
                "| Lambda | Original abs A bin | Token count | Zeroed rate | Retained L1 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for cal_lambda in LAMBDAS:
            for row in data["magnitude_profile"][str(cal_lambda)]["bins"]:
                upper = "inf" if not math.isfinite(row["upper"]) else f"{row['upper']:g}"
                lines.append(
                    f"| {cal_lambda} | [{row['lower']:g}, {upper}) | {row['count']:,} | "
                    f"{row['zeroed_ratio']:.2%} | {row['retained_l1_ratio']:.2%} |"
                )

        lines.extend(
            [
                "",
                "### Context classification for `1`, `2`, `-`, `=`",
                "",
                "| Token | Base-nonzero occurrences | Lambda | Zeroed | Zero rate | Top zeroed categories |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for symbol in SYMBOLS:
            detail = data["symbol_context"][symbol]
            for cal_lambda in LAMBDAS:
                metric = detail["per_lambda"][str(cal_lambda)]
                categories = sorted(
                    metric["zeroed_category_counts"].items(), key=lambda item: -item[1]
                )
                category_text = ", ".join(f"{name}={count:,}" for name, count in categories)
                lines.append(
                    f"| `{symbol}` | {detail['base_nonzero_occurrence_count']:,} | {cal_lambda} | "
                    f"{metric['zeroed_count']:,} | {metric['zeroed_ratio']:.2%} | {category_text} |"
                )

        lines.extend(
            [
                "",
                "### Top tokens by total L1 advantage removed",
                "",
                "| Lambda | Rank | Token | Total L1 removed | Changed occurrences |",
                "|---:|---:|---|---:|---:|",
            ]
        )
        for cal_lambda in LAMBDAS:
            for row in data["top_token_l1_changes"][str(cal_lambda)][:20]:
                token = str(row["token"]).replace("|", "\\|").replace("\n", "\\n")
                lines.append(
                    f"| {cal_lambda} | {row['rank']} | {token} | "
                    f"{row['total_l1_removed']:.3f} | {row['changed_occurrence_count']:,} |"
                )
        lines.append("")
    (SUMMARY_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    summary: dict[str, Any] = {"modes": {}}
    all_examples: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        print(f"ANALYZE_START {mode}", flush=True)
        mode_summary, examples = _analyze_mode(mode, tokenizer)
        summary["modes"][mode] = mode_summary
        all_examples[mode] = examples
        print(f"ANALYZE_DONE {mode}", flush=True)
    (SUMMARY_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (SUMMARY_DIR / "context_examples.json").write_text(
        json.dumps(all_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_coverage_csv(summary)
    _write_markdown(summary)
    print(SUMMARY_DIR / "summary.md")


if __name__ == "__main__":
    main()
