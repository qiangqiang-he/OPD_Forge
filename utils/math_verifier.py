"""Self-contained boxed-answer extraction and mathematical verification."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

from math_verify import parse, verify


_TEXT_LABELS = {
    "yes", "no", "true", "false", "a", "b", "c", "d", "e",
    "i", "ii", "iii", "iv", "v", "vi", "ap", "gp",
    "hyperbola", "parabola", "ellipse", "circle", "+", "-", "*", "/", "=",
}
_CHOICE_LIKE = {
    "a", "b", "c", "d", "e", "i", "ii", "iii", "iv", "v", "vi",
    "yes", "no", "true", "false", "ap", "gp", "+", "-", "*", "/", "=",
}


def _repair_common_latex_damage(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\x08oxed", r"\boxed")
    text = text.replace("\x0crac", r"\frac")
    text = text.replace("d\x0crac", r"\dfrac").replace("t\x0crac", r"\tfrac")
    return text.replace("\ufe68", "\\").replace("\uff3c", "\\")


def _strip_outer_math_wrappers(text: str) -> str:
    text = text.strip()
    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _is_balanced_braces(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _strip_one_outer_brace_pair(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == "{" and text[-1] == "}" and _is_balanced_braces(text[1:-1]):
        return text[1:-1].strip()
    return text


def _normalize_latex_fraction_shortcuts(text: str) -> str:
    text = re.sub(r"\\frac\s*([+-]?\d+)\s*\{([+-]?\d+)\}", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac\s*\{([+-]?\d+)\}\s*([+-]?\d+)", r"\\frac{\1}{\2}", text)
    return re.sub(r"\\frac\s*([+-]?\d+)\s*([+-]?\d+)", r"\\frac{\1}{\2}", text)


def _unwrap_latex_text_commands(text: str) -> str:
    def replace_match(match: re.Match) -> str:
        content = match.group(1).strip()
        choice = re.fullmatch(r"\(([A-Za-z])\)", content)
        return f" {choice.group(1) if choice else content} "

    return re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", replace_match, text)


def _normalize_math_string(text: str) -> str:
    text = _strip_outer_math_wrappers(_repair_common_latex_damage(text)).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")
    text = _normalize_latex_fraction_shortcuts(text.replace("∞", "\\infty"))
    text = _unwrap_latex_text_commands(text)
    text = re.sub(
        r"\\mbox\{\s*(?:inches?|inch|cm|centimeters?|units?)\s*\}\s*(?:\^\{?\d+\}?)?",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\^\\circ\b", "", text)
    text = re.sub(r"\bdegrees?\b|\bcents?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[，。.,;!?]+$", "", text.replace("\\$", ""))
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"\d{1,3}(?:,\s*\d{3})+(?:\.\d+)?", text):
        text = re.sub(r",\s*", "", text)
    return _strip_one_outer_brace_pair(text)


def _normalize_text_answer(text: str) -> str:
    text = _strip_outer_math_wrappers(_repair_common_latex_damage(text)).strip()
    text = _unwrap_latex_text_commands(text)
    text = re.sub(r"[，。.,;!?]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    choice = re.fullmatch(r"\(([a-z])\)", text)
    return choice.group(1) if choice else text


def extract_final_answer(response: str) -> tuple[str, bool]:
    """Extract the content of the last balanced ``\\boxed{...}``."""
    response = _repair_common_latex_damage(response)
    matches = list(re.finditer(r"\\boxed\s*\{", response))
    if not matches:
        return "", False
    brace_start = response.find("{", matches[-1].start())
    if brace_start == -1:
        return "", False
    index, depth = brace_start + 1, 1
    while index < len(response) and depth > 0:
        depth += (response[index] == "{") - (response[index] == "}")
        index += 1
    if depth != 0:
        return "", False
    content = response[brace_start + 1:index - 1].strip()
    return (_strip_outer_math_wrappers(content).strip(), True) if content else ("", False)


def _is_textual_answer(answer: str) -> bool:
    raw = _repair_common_latex_damage(answer).strip()
    if not raw:
        return False
    normalized = _normalize_text_answer(raw)
    if normalized in _TEXT_LABELS:
        return True
    if "\\" in raw or re.search(r"\d|[\^_{}\[\]()]", raw):
        return False
    if re.fullmatch(r"[A-Za-z]{1,3}", raw):
        return normalized in _CHOICE_LIKE or raw.isupper()
    return bool(re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){0,2}", raw))


def _try_parse_math(text: str):
    text = _normalize_math_string(text)
    seen = set()
    for candidate in (text, f"${text}$", f"\\boxed{{{text}}}"):
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = parse(candidate)
            if parsed:
                return parsed
        except Exception:
            pass
    return None


def _math_equivalent(prediction: str, target: str) -> bool:
    pred_norm, target_norm = _normalize_math_string(prediction), _normalize_math_string(target)
    if pred_norm.replace(" ", "") == target_norm.replace(" ", ""):
        return True
    # AMC-style JSON answers are decoded as floats (for example, ``36.0``),
    # while models normally emit the corresponding integer (``36``).  Do the
    # simple numeric comparison directly: math_verify's parser is not safe when
    # called from the reward manager's executor threads and silently fails here.
    numeric_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    if re.fullmatch(numeric_pattern, pred_norm) and re.fullmatch(numeric_pattern, target_norm):
        try:
            pred_decimal = Decimal(pred_norm)
            target_decimal = Decimal(target_norm)
            if pred_decimal.is_finite() and target_decimal.is_finite():
                return pred_decimal == target_decimal
        except InvalidOperation:
            pass
    parsed_target, parsed_pred = _try_parse_math(target_norm), _try_parse_math(pred_norm)
    if parsed_target is None or parsed_pred is None:
        return False
    try:
        return bool(verify(parsed_target, parsed_pred))
    except Exception:
        return False


def _answer_equivalent(prediction: str, target: str) -> bool:
    if not prediction or not target:
        return False
    pred_norm, target_norm = _normalize_math_string(prediction), _normalize_math_string(target)
    if pred_norm.replace(" ", "") == target_norm.replace(" ", ""):
        return True
    if _is_textual_answer(target):
        return _normalize_text_answer(prediction) == _normalize_text_answer(target)
    return _math_equivalent(prediction, target)


def verify_response_answer(response: str, answer: str) -> float:
    prediction, valid_format = extract_final_answer(response)
    if not valid_format:
        return 0.0
    return 1.0 if _answer_equivalent(prediction, str(answer)) else 0.0
