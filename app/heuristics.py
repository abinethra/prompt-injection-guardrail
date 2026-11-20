from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class HeuristicResult:
    score: float
    signals: List[str] = field(default_factory=list)


_OVERRIDE_PATTERNS: List[tuple[str, re.Pattern[str], float]] = [
    (
        "override_ignore_previous",
        re.compile(
            r"\b(ignore|disregard|forget|bypass|override)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,30}\b(instructions?|rules?|constraints?|directives?|prompt)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.45,
    ),
    (
        "override_new_instructions",
        re.compile(
            r"\b(new\s+(instructions?|rules?|directives?)|from\s+now\s+on|instead\s+of\s+that|do\s+not\s+follow)\b",
            re.IGNORECASE,
        ),
        0.30,
    ),
]

_EXTRACTION_PATTERNS: List[tuple[str, re.Pattern[str], float]] = [
    (
        "extract_system_prompt",
        re.compile(
            r"\b(reveal|show|print|dump|output|repeat|expose|leak|tell\s+me)\b.{0,35}\b(system\s+prompt|developer\s+message|hidden\s+prompt|initial\s+instructions?|secret\s+instructions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.45,
    ),
    (
        "extract_verbatim_prompt",
        re.compile(
            r"\b(verbatim|word\s+for\s+word|exact\s+text\s+of\s+your\s+(prompt|instructions?))\b",
            re.IGNORECASE,
        ),
        0.30,
    ),
]

_ROLEPLAY_PATTERNS: List[tuple[str, re.Pattern[str], float]] = [
    (
        "roleplay_dan_jailbreak",
        re.compile(
            r"\b(do\s+anything\s+now|\bdan\b|jailbreak|unrestricted\s+mode|developer\s+mode|god\s+mode|no\s+ethical\s+boundaries|no\s+restrictions)\b",
            re.IGNORECASE,
        ),
        0.40,
    ),
    (
        "roleplay_persona_shift",
        re.compile(
            r"\b(act\s+as|pretend\s+to\s+be|you\s+are\s+now|roleplay\s+as)\b.{0,45}\b(unfiltered|uncensored|evil|malicious|hacker|without\s+filters)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.35,
    ),
]

_BASE64_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])")
_LONG_ENCODED_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){12,}|(?:\\x[0-9A-Fa-f]{2}){8,}|[A-Za-z0-9_-]{64,}")
_DELIMITER_BLOCK_RE = re.compile(r"(?:```|===+|---+|###+|<<<+|>>>+|\[\[+|\]\]+|<\|im_(?:start|end)\|>|</?system>)", re.IGNORECASE)


def _has_decodable_base64_text(text: str) -> bool:
    for match in _BASE64_CANDIDATE_RE.finditer(text):
        token = match.group(0)
        padded = token + ("=" * ((4 - len(token) % 4) % 4))
        try:
            decoded = base64.b64decode(padded, validate=True)
            printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in decoded)
            if len(decoded) >= 16 and (printable / len(decoded)) >= 0.80:
                return True
        except Exception:
            continue
    return False


def analyze_heuristics(prompt: str) -> HeuristicResult:
    """Evaluate a prompt using deterministic security heuristics."""
    if not prompt or not prompt.strip():
        return HeuristicResult(score=0.0, signals=[])

    signals: List[str] = []
    raw_score = 0.0

    for name, pattern, weight in _OVERRIDE_PATTERNS:
        if pattern.search(prompt):
            signals.append(name)
            raw_score += weight

    for name, pattern, weight in _EXTRACTION_PATTERNS:
        if pattern.search(prompt):
            signals.append(name)
            raw_score += weight

    for name, pattern, weight in _ROLEPLAY_PATTERNS:
        if pattern.search(prompt):
            signals.append(name)
            raw_score += weight

    if _has_decodable_base64_text(prompt):
        signals.append("encoded_base64_payload")
        raw_score += 0.30
    elif _LONG_ENCODED_RE.search(prompt):
        signals.append("encoded_long_string")
        raw_score += 0.25

    delimiters = _DELIMITER_BLOCK_RE.findall(prompt)
    # Normalize delimiter count against prompt length so normal markdown code blocks don't over-trigger
    line_count = max(1, prompt.count("\n") + 1)
    delimiter_density = len(delimiters) / line_count
    if len(delimiters) >= 4 and delimiter_density >= 0.35:
        signals.append("high_delimiter_density")
        raw_score += 0.25

    # Smoothly map accumulated weight into [0.0, 1.0]
    bounded_score = round(1.0 - math.exp(-1.5 * raw_score), 4) if raw_score > 0 else 0.0
    bounded_score = min(1.0, max(0.0, bounded_score))

    return HeuristicResult(score=bounded_score, signals=signals)