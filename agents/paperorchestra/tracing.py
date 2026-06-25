"""Prompt tracing and checkpoint helpers for PaperOrchestra."""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

from agents.llm_client import call_llm, parse_llm_json_text


class PaperGenerationTrace:
    """Append-only JSONL trace for manuscript generation stages."""

    def __init__(self, path: Path | None):
        self.path = path
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, status: str, **fields: Any) -> None:
        if not self.path:
            return
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "status": status,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _prompt_stats(system_prompt: str, user_prompt: str) -> dict[str, int]:
    system_chars = len(system_prompt or "")
    user_chars = len(user_prompt or "")
    # Conservative proxy for BPE-ish text; enough for budget monitoring.
    estimated_tokens = max(1, (system_chars + user_chars) // 4)
    return {
        "system_chars": system_chars,
        "user_chars": user_chars,
        "total_chars": system_chars + user_chars,
        "estimated_tokens": estimated_tokens,
    }


class _DeadlineExceeded(TimeoutError):
    pass


def _call_llm_with_deadline(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    max_tokens: int | None,
    timeout_seconds: int | None,
) -> tuple[str, int]:
    """Run a single LLM call with an optional wall-clock deadline."""
    if not timeout_seconds or timeout_seconds <= 0:
        return call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        return call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)

    old_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise _DeadlineExceeded(f"LLM call exceeded {timeout_seconds}s deadline")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def call_text_traced(
    stage: str,
    system_prompt: str,
    user_prompt: str,
    *,
    trace: PaperGenerationTrace | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    fallback_user_prompts: list[str] | None = None,
    min_chars: int = 10,
    timeout_seconds: int | None = None,
) -> tuple[str, int]:
    """Call the text LLM with trace rows and smaller prompt fallbacks."""
    prompts = [user_prompt] + list(fallback_user_prompts or [])
    last_exc: Exception | None = None
    for tier, candidate_user in enumerate(prompts, start=1):
        stats = _prompt_stats(system_prompt, candidate_user)
        start = time.time()
        if trace:
            trace.log(stage, "started", fallback_tier=tier, **stats)
        try:
            text, tokens = _call_llm_with_deadline(
                system_prompt,
                candidate_user,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            latency = time.time() - start
            if not text or len(text.strip()) < min_chars:
                raise RuntimeError(f"empty_or_short_response:{len(text or '')}")
            if trace:
                trace.log(
                    stage,
                    "ok",
                    fallback_tier=tier,
                    latency_seconds=round(latency, 3),
                    response_chars=len(text),
                    tokens=tokens,
                    **stats,
                )
            return text, tokens
        except Exception as exc:  # noqa: BLE001
            latency = time.time() - start
            last_exc = exc
            if trace:
                trace.log(
                    stage,
                    "error",
                    fallback_tier=tier,
                    latency_seconds=round(latency, 3),
                    error=str(exc),
                    **stats,
                )
    assert last_exc is not None
    raise last_exc


def call_json_traced(
    stage: str,
    system_prompt: str,
    user_prompt: str,
    *,
    trace: PaperGenerationTrace | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    fallback_user_prompts: list[str] | None = None,
    timeout_seconds: int | None = None,
) -> tuple[dict | list, int]:
    """Call the LLM and parse JSON with trace rows and prompt fallbacks."""
    prompts = [user_prompt] + list(fallback_user_prompts or [])
    last_exc: Exception | None = None
    for tier, candidate_user in enumerate(prompts, start=1):
        stats = _prompt_stats(system_prompt, candidate_user)
        start = time.time()
        if trace:
            trace.log(stage, "started", fallback_tier=tier, **stats)
        try:
            text, tokens = _call_llm_with_deadline(
                system_prompt,
                candidate_user,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            if not text or not text.strip():
                raise RuntimeError("empty_response")
            parsed, method = parse_llm_json_text(text)
            latency = time.time() - start
            if trace:
                trace.log(
                    stage,
                    "ok",
                    fallback_tier=tier,
                    latency_seconds=round(latency, 3),
                    response_chars=len(text),
                    tokens=tokens,
                    parse_method=method,
                    **stats,
                )
            return parsed, tokens
        except Exception as exc:  # noqa: BLE001
            latency = time.time() - start
            last_exc = exc
            if trace:
                trace.log(
                    stage,
                    "error",
                    fallback_tier=tier,
                    latency_seconds=round(latency, 3),
                    error=str(exc),
                    **stats,
                )
    assert last_exc is not None
    raise last_exc


def checkpoint_path(root: Path, name: str) -> Path:
    return root / "paperorchestra_checkpoints" / name


def read_json_checkpoint(root: Path, name: str) -> Any | None:
    path = checkpoint_path(root, name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json_checkpoint(root: Path, name: str, payload: Any) -> None:
    path = checkpoint_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_text_checkpoint(root: Path, name: str) -> str | None:
    path = checkpoint_path(root, name)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text if text.strip() else None


def write_text_checkpoint(root: Path, name: str, payload: str) -> None:
    path = checkpoint_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload or "", encoding="utf-8")


def get_or_create_checkpoint(
    root: Path,
    name: str,
    producer: Callable[[], Any],
    *,
    text: bool = False,
) -> Any:
    cached = read_text_checkpoint(root, name) if text else read_json_checkpoint(root, name)
    if cached is not None:
        return cached
    value = producer()
    if text:
        write_text_checkpoint(root, name, str(value or ""))
    else:
        write_json_checkpoint(root, name, value)
    return value
