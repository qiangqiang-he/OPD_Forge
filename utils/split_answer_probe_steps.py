"""Compare deterministic step splitters and save answer-probe step spans."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


PROTECTED_BEGIN_RE = re.compile(
    r"\\begin\{(?:equation|equation\*|align|align\*|aligned|gather|gather\*|multline|multline\*|cases)\}"
)
PROTECTED_END_RE = re.compile(
    r"\\end\{(?:equation|equation\*|align|align\*|aligned|gather|gather\*|multline|multline\*|cases)\}"
)
HR_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
EXPLICIT_STEP_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?(?:✅\s*)?"
    r"(?:step|part|case|subcase|solution\s+step|步骤|第\s*[一二三四五六七八九十\d]+\s*步)"
    r"\s*(?:#?\s*)?(?:\d+|[ivxlcdm]+)?\b",
    re.IGNORECASE,
)
FINAL_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?(?:✅\s*)?"
    r"(?:final\s+(?:answer|result|conclusion)|conclusion|answer|最终答案|结论)\b",
    re.IGNORECASE,
)
CONNECTOR_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?"
    r"(?:so|thus|therefore|hence|consequently|accordingly|this\s+(?:gives|yields|means|shows)|"
    r"substitut(?:e|ing)|simplif(?:y|ying)|evaluat(?:e|ing)|combining|it\s+follows|"
    r"now\s+(?:we\s+)?(?:have|obtain)|所以|因此|故|于是|代入|化简)\b",
    re.IGNORECASE,
)
INTRO_END_RE = re.compile(r"(?:[:：]|(?:as follows)|(?:we (?:get|have|obtain)))[\s*]*$", re.IGNORECASE)
LIST_LINE_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+\S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("../FrugalRL/models/Qwen3-1.7B"))
    return parser.parse_args()


def _trimmed_span(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end, text[start:end]) if start < end else None


def _line_state_transitions(line: str, state: str | None) -> str | None:
    stripped = line.strip()
    if state in {"fence_backtick", "fence_tilde"}:
        marker = "```" if state == "fence_backtick" else "~~~"
        return None if stripped.startswith(marker) else state
    if state == "dollar_math":
        return None if len(re.findall(r"(?<!\\)\$\$", line)) % 2 else state
    if state == "bracket_math":
        return None if r"\]" in line else state
    if state == "latex_env":
        return None if PROTECTED_END_RE.search(line) else state

    if stripped.startswith("```"):
        return "fence_backtick"
    if stripped.startswith("~~~"):
        return "fence_tilde"
    if len(re.findall(r"(?<!\\)\$\$", line)) % 2:
        return "dollar_math"
    if r"\[" in line and r"\]" not in line[line.find(r"\[") + 2 :]:
        return "bracket_math"
    if PROTECTED_BEGIN_RE.search(line) and not PROTECTED_END_RE.search(line):
        return "latex_env"
    return None


def protected_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    state = None
    range_start = None
    cursor = 0
    for line in text.splitlines(keepends=True):
        old_state = state
        state = _line_state_transitions(line, state)
        if old_state is None and state is not None:
            range_start = cursor
        if old_state is not None and state is None and range_start is not None:
            ranges.append((range_start, cursor + len(line.rstrip("\r\n"))))
            range_start = None
        cursor += len(line)
    if state is not None and range_start is not None:
        ranges.append((range_start, len(text)))
    return ranges


def safe_markdown_blocks(text: str) -> list[Span]:
    """Split on blank lines, except inside fenced code and display math."""
    spans = []
    state = None
    block_start = 0
    cursor = 0
    blank_start = None
    for line in text.splitlines(keepends=True):
        old_state = state
        state = _line_state_transitions(line, state)
        is_unprotected_blank = old_state is None and state is None and not line.strip()
        if is_unprotected_blank:
            if blank_start is None:
                blank_start = cursor
        else:
            if blank_start is not None:
                span = _trimmed_span(text, block_start, blank_start)
                if span:
                    spans.append(span)
                block_start = cursor
                blank_start = None
        cursor += len(line)
    boundary = blank_start if blank_start is not None else len(text)
    span = _trimmed_span(text, block_start, boundary)
    if span:
        spans.append(span)
    return spans


def split_nonempty_lines(text: str) -> list[Span]:
    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        span = _trimmed_span(text, cursor, cursor + len(line))
        if span:
            spans.append(span)
        cursor += len(line)
    if cursor < len(text):
        span = _trimmed_span(text, cursor, len(text))
        if span:
            spans.append(span)
    return spans


def split_blank_lines(text: str) -> list[Span]:
    spans = []
    cursor = 0
    for match in re.finditer(r"\n[ \t]*\n+", text):
        span = _trimmed_span(text, cursor, match.start())
        if span:
            spans.append(span)
        cursor = match.end()
    span = _trimmed_span(text, cursor, len(text))
    if span:
        spans.append(span)
    return spans


def _first_line(span: Span) -> str:
    """Return the first substantive line in a source-preserving span."""

    return next((line.strip() for line in span.text.splitlines() if line.strip()), "")


def _is_math_only(span: Span) -> bool:
    value = span.text.strip()
    return (
        (value.startswith("$$") and value.endswith("$$"))
        or (value.startswith(r"\[") and value.endswith(r"\]"))
        or (PROTECTED_BEGIN_RE.match(value) is not None and PROTECTED_END_RE.search(value) is not None)
    )


def _starts_with_display_math(span: Span) -> bool:
    value = span.text.lstrip()
    return bool(
        value.startswith("$$")
        or value.startswith(r"\[")
        or PROTECTED_BEGIN_RE.match(value)
    )


def _is_list(span: Span) -> bool:
    lines = [line for line in span.text.splitlines() if line.strip()]
    return bool(lines) and sum(bool(LIST_LINE_RE.match(line)) for line in lines) >= max(1, math.ceil(len(lines) / 2))


def _is_heading(span: Span) -> bool:
    return HEADING_RE.match(_first_line(span)) is not None


def _is_explicit_step(span: Span) -> bool:
    return EXPLICIT_STEP_RE.match(_first_line(span)) is not None


def _is_final_heading(span: Span) -> bool:
    return FINAL_HEADING_RE.match(_first_line(span)) is not None


def _join_spans(text: str, spans: list[Span]) -> Span:
    return Span(spans[0].start, spans[-1].end, text[spans[0].start : spans[-1].end])


def _split_blocks_at_strong_boundaries(text: str, blocks: list[Span]) -> list[Span]:
    """Split Markdown blocks at authored headings/steps, never inside protection.

    Models often omit a blank line before ``Step 2`` (or insert blank lines
    everywhere else).  Consequently blank lines are only candidate paragraph
    boundaries; an authored marker at the beginning of a physical line is the
    reliable boundary.  ``safe_markdown_blocks`` has already guaranteed that a
    protected display/code region remains in one block.
    """

    results: list[Span] = []
    for block in blocks:
        state = None
        chunk_start = block.start
        cursor = block.start
        for line in block.text.splitlines(keepends=True):
            old_state = state
            state = _line_state_transitions(line, state)
            stripped_span = _trimmed_span(text, cursor, cursor + len(line))
            is_protected = old_state is not None or state is not None
            is_boundary = bool(
                stripped_span
                and not is_protected
                and (
                    HEADING_RE.match(stripped_span.text)
                    or EXPLICIT_STEP_RE.match(stripped_span.text)
                    or FINAL_HEADING_RE.match(stripped_span.text)
                )
            )
            if is_boundary and cursor > chunk_start:
                prefix = _trimmed_span(text, chunk_start, cursor)
                if prefix:
                    results.append(prefix)
                chunk_start = cursor
            cursor += len(line)
        suffix = _trimmed_span(text, chunk_start, block.end)
        if suffix:
            results.append(suffix)
    return results


def _is_tiny_fragment(span: Span) -> bool:
    compact = re.sub(r"\s+", " ", span.text).strip()
    if not compact:
        return True
    alphanumeric = re.sub(r"[^\w]+", "", compact, flags=re.UNICODE)
    return len(alphanumeric) < 24 or len(compact.split()) <= 3


def _merge_tiny_groups(text: str, groups: list[list[Span]]) -> list[list[Span]]:
    """Prevent labels, punctuation, and short bridge text becoming steps."""

    if len(groups) <= 1:
        return groups
    merged: list[list[Span]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        joined = _join_spans(text, group)
        authored_boundary = (
            _is_heading(joined)
            or _is_explicit_step(joined)
            or _is_final_heading(joined)
        )
        if _is_tiny_fragment(joined) and not authored_boundary:
            # A connector concludes the preceding derivation.  Other tiny
            # fragments (notably an introductory "Solution:") are safer when
            # attached to what follows.
            if merged and CONNECTOR_RE.match(_first_line(joined)):
                merged[-1].extend(group)
            elif index + 1 < len(groups):
                groups[index + 1] = group + groups[index + 1]
            elif merged:
                merged[-1].extend(group)
            else:
                merged.append(group)
        else:
            merged.append(group)
        index += 1
    return merged


def _partition_source_spans(text: str, spans: list[Span]) -> list[Span]:
    """Turn semantic spans into an exact, non-overlapping source partition.

    Inter-step whitespace and Markdown separators are assigned to the previous
    step.  This makes concatenating ``response[char_start:char_end]`` recover
    the original response exactly and gives token mapping one unambiguous,
    monotone sequence of boundaries.
    """

    if not spans:
        return []
    boundaries = [0, *[span.start for span in spans[1:]], len(text)]
    return [
        Span(boundaries[index], boundaries[index + 1], text[boundaries[index] : boundaries[index + 1]])
        for index in range(len(boundaries) - 1)
        if boundaries[index] < boundaries[index + 1]
    ]


def _split_markdown_attachment(text: str, blocks: list[Span] | None = None) -> list[Span]:
    """Attach display math/lists to prose without treating blank lines as steps."""
    blocks = blocks if blocks is not None else safe_markdown_blocks(text)
    blocks = _split_blocks_at_strong_boundaries(text, blocks)
    groups: list[list[Span]] = []
    current: list[Span] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for block in blocks:
        if HR_RE.match(block.text):
            # A thematic break is a boundary hint, but never a standalone
            # semantic step.  Its source characters are recovered later by
            # _partition_source_spans.
            flush()
            continue
        first = _first_line(block)
        if _is_heading(block) or _is_explicit_step(block) or _is_final_heading(block):
            flush()
            current = [block]
        elif _is_math_only(block) or _starts_with_display_math(block):
            if current:
                current.append(block)
            else:
                current = [block]
        elif _is_list(block):
            if current and (INTRO_END_RE.search(current[-1].text.rstrip()) or _is_heading(current[0])):
                current.append(block)
            else:
                flush()
                current = [block]
        elif not current:
            current = [block]
        elif (
            (len(current) == 1 and (_is_heading(current[0]) or _is_explicit_step(current[0])))
            or INTRO_END_RE.search(current[-1].text.rstrip())
            or _is_tiny_fragment(_join_spans(text, current))
        ):
            current.append(block)
        elif CONNECTOR_RE.match(first):
            current.append(block)
        elif current and CONNECTOR_RE.match(_first_line(current[-1])):
            current.append(block)
        elif _is_tiny_fragment(block):
            current.append(block)
        else:
            flush()
            current = [block]
    flush()
    groups = _merge_tiny_groups(text, groups)
    return [_join_spans(text, group) for group in groups]


def split_markdown_attachment(text: str) -> list[Span]:
    return _partition_source_spans(text, _split_markdown_attachment(text))


def split_semantic_hybrid(text: str) -> list[Span]:
    """Return stable, source-preserving semantic reasoning steps.

    Protection is applied before any boundary discovery.  Headings and
    explicit Step/Part/Case/Subcase markers are strong boundaries even when a
    model forgot the surrounding blank lines.  In their absence, Markdown
    paragraphs are merely candidates and are merged with formulas, connective
    conclusions, lists, and short fragments.
    """

    if not text or not text.strip():
        return []
    blocks = safe_markdown_blocks(text)
    chunks = _split_blocks_at_strong_boundaries(text, blocks)
    marker_positions = [
        index
        for index, chunk in enumerate(chunks)
        if _is_heading(chunk) or _is_explicit_step(chunk) or _is_final_heading(chunk)
    ]
    if not marker_positions:
        results = _split_markdown_attachment(text, blocks)
    else:
        results = []
        first_marker = marker_positions[0]
        if first_marker:
            results.extend(_split_markdown_attachment(text, chunks[:first_marker]))
        for marker_offset, marker_index in enumerate(marker_positions):
            next_marker = (
                marker_positions[marker_offset + 1]
                if marker_offset + 1 < len(marker_positions)
                else len(chunks)
            )
            # An authored Markdown section is already a semantic unit.  Blank
            # lines inside it are formatting, not additional reasoning steps.
            results.append(_join_spans(text, chunks[marker_index:next_marker]))

    # Very long authored sections are re-run through their internal Markdown
    # candidates.  A single indivisible block is deliberately retained rather
    # than cutting a formula or inventing line-based reasoning steps.
    refined: list[Span] = []
    for result in results:
        if len(result.text) <= 3500:
            refined.append(result)
            continue
        candidates = safe_markdown_blocks(result.text)
        if len(candidates) <= 1:
            refined.append(result)
            continue
        absolute_candidates = [
            Span(result.start + item.start, result.start + item.end, item.text)
            for item in candidates
        ]
        decomposed = _split_markdown_attachment(text, absolute_candidates)
        refined.extend(decomposed if len(decomposed) > 1 else [result])

    return _partition_source_spans(text, refined)


METHODS: dict[str, Callable[[str], list[Span]]] = {
    "nonempty_line": split_nonempty_lines,
    "blank_line": split_blank_lines,
    "markdown_attachment": split_markdown_attachment,
    "semantic_hybrid": split_semantic_hybrid,
}


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _validate_spans(text: str, spans: list[Span]) -> None:
    if not spans:
        raise ValueError("A non-empty response produced no steps.")
    previous_end = 0
    covered = [False] * len(text)
    for span in spans:
        if span.start < previous_end or span.start >= span.end:
            raise ValueError(f"Invalid or overlapping span: {span}")
        if text[span.start : span.end] != span.text:
            raise ValueError("Step text does not match its source span.")
        for index in range(span.start, span.end):
            covered[index] = True
        previous_end = span.end
    remainder = "".join(character for index, character in enumerate(text) if not covered[index])
    remainder = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", remainder)
    if remainder.strip():
        raise ValueError(f"Splitter dropped substantive text: {remainder[:200]!r}")


def _step_kind(span: Span) -> str:
    if _is_explicit_step(span):
        return "explicit_step"
    if _is_final_heading(span):
        return "final"
    if _is_heading(span):
        return "heading_section"
    if _is_math_only(span):
        return "math"
    if _is_list(span):
        return "list"
    return "prose_or_mixed"


def _select_audit_indices(records: list[dict[str, Any]]) -> list[int]:
    features = []
    for index, record in enumerate(records):
        text = record["response"]
        blocks = safe_markdown_blocks(text)
        features.append(
            {
                "index": index,
                "length": record["num_tokens"],
                "math": sum(_is_math_only(block) for block in blocks),
                "explicit": sum(_is_explicit_step(block) for block in blocks),
            }
        )
    chosen = []
    rankings = [
        sorted(features, key=lambda row: row["length"]),
        sorted(features, key=lambda row: row["length"], reverse=True),
        sorted(features, key=lambda row: row["math"], reverse=True),
        sorted(features, key=lambda row: (row["explicit"] > 0, row["length"]), reverse=True),
        sorted(features, key=lambda row: (row["explicit"] == 0, row["length"]), reverse=True),
    ]
    for ranking in rankings:
        for item in ranking[:8]:
            if item["index"] not in chosen:
                chosen.append(item["index"])
                break
    stride = max(1, len(records) // 11)
    for index in range(0, len(records), stride):
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= 16:
            break
    return chosen


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer.resolve(), trust_remote_code=True)

    method_outputs: dict[str, list[list[dict[str, Any]]]] = {method: [] for method in METHODS}
    method_token_counts: dict[str, list[int]] = {method: [] for method in METHODS}
    report: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "num_rollouts": len(records),
        "methods": {},
    }

    for method_name, splitter in METHODS.items():
        response_step_counts = []
        orphan_math_count = 0
        heading_only_count = 0
        incomplete_intro_count = 0
        protected_boundary_count = 0
        for record in records:
            response = record["response"]
            spans = splitter(response)
            _validate_spans(response, spans)
            protected = protected_ranges(response)
            step_rows = []
            for step_index, span in enumerate(spans):
                token_count = len(tokenizer.encode(span.text, add_special_tokens=False))
                kind = _step_kind(span)
                step_rows.append(
                    {
                        "step_index": step_index,
                        "char_start": span.start,
                        "char_end": span.end,
                        "num_tokens": token_count,
                        "kind": kind,
                        "text": span.text,
                    }
                )
                method_token_counts[method_name].append(token_count)
                orphan_math_count += int(kind == "math")
                heading_only_count += int(HEADING_RE.match(span.text) is not None and "\n" not in span.text)
                incomplete_intro_count += int(INTRO_END_RE.search(span.text.rstrip()) is not None)
            for boundary in [span.end for span in spans[:-1]]:
                protected_boundary_count += sum(start < boundary < end for start, end in protected)
            response_step_counts.append(len(spans))
            method_outputs[method_name].append(step_rows)

        token_counts = method_token_counts[method_name]
        report["methods"][method_name] = {
            "total_steps": len(token_counts),
            "mean_steps_per_response": statistics.fmean(response_step_counts),
            "median_steps_per_response": statistics.median(response_step_counts),
            "p10_steps_per_response": _quantile(response_step_counts, 0.10),
            "p90_steps_per_response": _quantile(response_step_counts, 0.90),
            "mean_tokens_per_step": statistics.fmean(token_counts),
            "median_tokens_per_step": statistics.median(token_counts),
            "p10_tokens_per_step": _quantile(token_counts, 0.10),
            "p90_tokens_per_step": _quantile(token_counts, 0.90),
            "steps_le_8_tokens_rate": sum(value <= 8 for value in token_counts) / len(token_counts),
            "steps_gt_768_tokens_rate": sum(value > 768 for value in token_counts) / len(token_counts),
            "orphan_math_step_rate": orphan_math_count / len(token_counts),
            "heading_only_step_rate": heading_only_count / len(token_counts),
            "intro_ending_step_rate": incomplete_intro_count / len(token_counts),
            "boundaries_inside_protected_markdown": protected_boundary_count,
        }

        output_lines = []
        for record, steps in zip(records, method_outputs[method_name], strict=True):
            output_lines.append(
                json.dumps(
                    {
                        "selection_index": record["selection_index"],
                        "question_index": record["question_index"],
                        "sample_index": record["sample_index"],
                        "method": method_name,
                        "num_steps": len(steps),
                        "steps": steps,
                    },
                    ensure_ascii=False,
                )
            )
        _atomic_write(args.output_dir / f"steps_{method_name}.jsonl", "\n".join(output_lines) + "\n")

    _atomic_write(args.output_dir / "segmentation_report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    audit_lines = ["# Semantic-hybrid segmentation audit", ""]
    for record_index in _select_audit_indices(records):
        record = records[record_index]
        audit_lines.extend(
            [
                f"## selection={record['selection_index']} sample={record['sample_index']} "
                f"question_index={record['question_index']} response_tokens={record['num_tokens']}",
                "",
                f"Question: {record['question']}",
                "",
            ]
        )
        for step in method_outputs["semantic_hybrid"][record_index]:
            audit_lines.extend(
                [
                    f"### Probe step {step['step_index']} ({step['kind']}, {step['num_tokens']} tokens)",
                    "",
                    step["text"],
                    "",
                ]
            )
    _atomic_write(args.output_dir / "semantic_hybrid_audit.md", "\n".join(audit_lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
