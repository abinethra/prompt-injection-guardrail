from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.detector import GuardrailDetector

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv_if_present(env_path: Path = ROOT_DIR / ".env") -> None:
    """Lightweight .env loader so local development works without extra dependencies."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv_if_present()


def _parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(val: Optional[str], default: float) -> float:
    if val is None or not val.strip():
        return default
    return float(val.strip())


FLAG_THRESHOLD = _parse_float(os.environ.get("GUARDRAIL_FLAG_THRESHOLD"), 0.45)
BLOCK_THRESHOLD = _parse_float(os.environ.get("GUARDRAIL_BLOCK_THRESHOLD"), 0.70)
USE_GEMINI_API = _parse_bool(os.environ.get("USE_GEMINI_API"), default=False)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
LOG_FILE_PATH = Path(os.environ.get("LOG_FILE_PATH", str(ROOT_DIR / "logs" / "decisions.jsonl")))
if not LOG_FILE_PATH.is_absolute():
    LOG_FILE_PATH = ROOT_DIR / LOG_FILE_PATH


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000, description="User prompt to screen and answer.")
    session_id: Optional[str] = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    decision: Literal["allowed", "flagged", "blocked"]
    score: float
    signals: List[str]
    latency_ms: float
    reply: Optional[str] = None
    output_redacted: bool = False
    output_warnings: List[str] = Field(default_factory=list)
    provider: str = "mock_llm"


class HealthResponse(BaseModel):
    status: str
    gemini_enabled: bool
    flag_threshold: float
    block_threshold: float
    log_file: str


class StatsResponse(BaseModel):
    total_requests: int
    allowed: int
    flagged: int
    blocked: int
    avg_score: float
    avg_latency_ms: float
    top_signals: dict[str, int]


_SECRET_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("openai_or_stripe_key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|rk_(?:live|test)_[A-Za-z0-9]{16,})\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("email_address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("system_prompt_leak_phrase", re.compile(r"\bmy system prompt is\b", re.IGNORECASE)),
]


def scan_and_sanitize_output(text: str) -> tuple[str, bool, List[str]]:
    """Scan LLM output for secret-like strings and redact matches if found."""
    sanitized = text
    warnings: List[str] = []
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(sanitized):
            warnings.append(label)
            sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized, bool(warnings), warnings


def call_mock_llm(prompt: str) -> tuple[str, str]:
    """Deterministic local mock LLM for offline/default proxy testing."""
    cleaned = " ".join(prompt.strip().split())
    preview = cleaned[:120] + ("..." if len(cleaned) > 120 else "")
    return f"[MockLLM] Acknowledged safe request: {preview}", "mock_llm"


def call_gemini_llm(prompt: str) -> tuple[str, str]:
    """Optional Gemini REST API call when USE_GEMINI_API=true and GEMINI_API_KEY is set."""
    if not GEMINI_API_KEY:
        raise RuntimeError("USE_GEMINI_API is enabled, but GEMINI_API_KEY is missing.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ]
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
    except urllib_error.HTTPError as exc:
        raise RuntimeError(f"Gemini API HTTP error {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    candidates = parsed.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini API returned no candidates.")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text_chunks = [str(p["text"]) for p in parts if isinstance(p, dict) and "text" in p]
    if not text_chunks:
        raise RuntimeError("Gemini API response contained no text.")
    return "\n".join(text_chunks).strip(), f"gemini:{GEMINI_MODEL}"


def generate_llm_reply(prompt: str) -> tuple[str, str]:
    if USE_GEMINI_API:
        return call_gemini_llm(prompt)
    return call_mock_llm(prompt)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def append_decision_log(
    log_path: Path,
    decision: str,
    score: float,
    signals: List[str],
    latency_ms: float,
    prompt: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "score": round(float(score), 4),
        "signals": signals,
        "latency_ms": round(float(latency_ms), 4),
        "prompt_hash": hash_prompt(prompt),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def classify_decision(score: float) -> Literal["allowed", "flagged", "blocked"]:
    if score >= BLOCK_THRESHOLD:
        return "blocked"
    if score >= FLAG_THRESHOLD:
        return "flagged"
    return "allowed"


def get_detector() -> GuardrailDetector:
    detector = getattr(app.state, "detector", None)
    if detector is None:
        detector = GuardrailDetector(threshold=FLAG_THRESHOLD)
        app.state.detector = detector
    return detector


app = FastAPI(title="Prompt Injection & Jailbreak Guardrail Proxy", version="0.4.0")


@app.on_event("startup")
def on_startup() -> None:
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE_PATH.exists():
        LOG_FILE_PATH.touch()
    # Initialize detector lazily if model files aren't built yet during isolated tests
    try:
        app.state.detector = GuardrailDetector(threshold=FLAG_THRESHOLD)
    except FileNotFoundError:
        app.state.detector = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        gemini_enabled=USE_GEMINI_API and bool(GEMINI_API_KEY),
        flag_threshold=FLAG_THRESHOLD,
        block_threshold=BLOCK_THRESHOLD,
        log_file=str(LOG_FILE_PATH),
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    if not LOG_FILE_PATH.exists():
        return StatsResponse(
            total_requests=0,
            allowed=0,
            flagged=0,
            blocked=0,
            avg_score=0.0,
            avg_latency_ms=0.0,
            top_signals={},
        )

    total = 0
    allowed = 0
    flagged = 0
    blocked = 0
    score_sum = 0.0
    latency_sum = 0.0
    signal_counts: dict[str, int] = {}

    for line in LOG_FILE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        total += 1
        dec = row.get("decision")
        if dec == "allowed":
            allowed += 1
        elif dec == "flagged":
            flagged += 1
        elif dec == "blocked":
            blocked += 1

        score_sum += float(row.get("score", 0.0))
        latency_sum += float(row.get("latency_ms", 0.0))
        for sig in row.get("signals") or []:
            signal_counts[str(sig)] = signal_counts.get(str(sig), 0) + 1

    top_sorted = dict(sorted(signal_counts.items(), key=lambda kv: kv[1], reverse=True))
    return StatsResponse(
        total_requests=total,
        allowed=allowed,
        flagged=flagged,
        blocked=blocked,
        avg_score=round(score_sum / total, 4) if total else 0.0,
        avg_latency_ms=round(latency_sum / total, 4) if total else 0.0,
        top_signals=top_sorted,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()

    try:
        detector = get_detector()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    detection = detector.score_prompt(req.prompt, threshold=FLAG_THRESHOLD)
    decision = classify_decision(detection.score)

    reply_text: Optional[str] = None
    redacted = False
    warnings: List[str] = []
    provider = "none"

    if decision == "blocked":
        reply_text = "Request blocked by security guardrail policy."
        provider = "guardrail"
    else:
        # Both 'allowed' and 'flagged' (low-confidence borderline) proceed with output screening
        try:
            raw_reply, provider = generate_llm_reply(req.prompt)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        reply_text, redacted, warnings = scan_and_sanitize_output(raw_reply)
        if redacted:
            warnings.append("output_secret_redacted")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    all_signals = list(dict.fromkeys(detection.signals + warnings))

    append_decision_log(
        log_path=LOG_FILE_PATH,
        decision=decision,
        score=detection.score,
        signals=all_signals,
        latency_ms=elapsed_ms,
        prompt=req.prompt,
    )

    return ChatResponse(
        decision=decision,
        score=detection.score,
        signals=all_signals,
        latency_ms=round(elapsed_ms, 4),
        reply=reply_text,
        output_redacted=redacted,
        output_warnings=warnings,
        provider=provider,
    )