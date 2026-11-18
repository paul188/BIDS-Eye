"""
LLM_preprocessor/llm_client.py
------------------------------
One shared LLM call path for the whole text-to-SQL pipeline, replacing the three
divergent Gemini cascades in preprocess/ and services/translate/text_to_sql.py.

`llm_generate()` walks a configurable tier list and returns the first successful
text response:

    gemini-2.5-flash -> gemini-2.5-pro -> gemini-2.5-flash-lite -> gemini-2.0-flash
        -> claude-sonnet-4-6   (Anthropic, cross-provider last resort)

Resilience rules:
  * Retry the SAME tier (with backoff) only on TRANSIENT errors
    (Gemini 503/429/UNAVAILABLE/RESOURCE_EXHAUSTED/DEADLINE/5xx; Anthropic
    RateLimit/Overloaded/InternalServer/Connection/Timeout). On a non-transient
    error (400/401/403) skip straight to the next tier — no wasted retries.
  * `temperature` is passed only to tiers that accept it. Claude Opus 4.8/4.7 and
    Fable 5 reject `temperature` (400), so the default Claude tier is
    `claude-sonnet-4-6`, which accepts it.
  * If every tier fails, raise `LLMAllFailedError` — callers decide whether to
    degrade gracefully (see services/translate/text_to_sql.py).

Config (env):
  GEMINI_API_KEY / GOOGLE_API_KEY  — Gemini key (also accepted as a call arg)
  ANTHROPIC_API_KEY                — Claude key; if absent, the Claude tier is skipped
  GEMINI_MODEL_CASCADE             — comma-separated override of the Gemini tier order
  ANTHROPIC_FALLBACK_MODEL         — override the Claude model (default claude-sonnet-4-6)
"""

from __future__ import annotations

import json
import logging
import contextvars
import os
import threading
import time
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)

# Claude models that reject the `temperature` sampling param (HTTP 400).
_NO_TEMPERATURE_PREFIXES = ("claude-opus-4-8", "claude-opus-4-7", "claude-fable-5")

_DEFAULT_GEMINI_CASCADE = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]
# Haiku is the fallback model on purpose: this tier only fires when Gemini is
# down, and it runs for BOTH the intent and SQL-generation stages, so speed
# matters more than peak quality. Sonnet took ~30s on the large SQL prompt and
# blew past Cloudflare's ~100s request limit; Haiku answers the same prompt in a
# few seconds. Override with ANTHROPIC_FALLBACK_MODEL if you want Sonnet/Opus.
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"


class LLMAllFailedError(RuntimeError):
    """Raised when every tier in the cascade fails."""


# ── token usage ─────────────────────────────────────────────────────────────────
# Every successful call is logged with its token counts, and added to the list
# started by start_usage_collection() when there is one, so the cost of one
# search can be summed per stage. Failed attempts report nothing.

_usage_collector: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "llm_usage_collector", default=None)
_last_call = threading.local()


def start_usage_collection() -> list:
    """Collect the usage of every later call in this context; returns the list.

    Each entry: purpose, provider, model, input_tokens (cached ones included),
    cached_tokens, output_tokens, thinking_tokens, seconds.
    """
    collected: list = []
    _usage_collector.set(collected)
    return collected


def _count(obj, name: str) -> int:
    value = getattr(obj, name, None)
    return value if isinstance(value, int) else 0


def _note_gemini_usage(resp) -> None:
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return
    _last_call.usage = {
        "input_tokens": _count(meta, "prompt_token_count"),
        "cached_tokens": _count(meta, "cached_content_token_count"),
        "output_tokens": _count(meta, "candidates_token_count"),
        "thinking_tokens": _count(meta, "thoughts_token_count"),
    }


def _note_anthropic_usage(resp) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    cached = _count(usage, "cache_read_input_tokens")
    _last_call.usage = {
        # Anthropic reports uncached, cache-read and cache-write input apart.
        "input_tokens": (_count(usage, "input_tokens") + cached
                         + _count(usage, "cache_creation_input_tokens")),
        "cached_tokens": cached,
        "output_tokens": _count(usage, "output_tokens"),
        "thinking_tokens": 0,
    }


def _report_usage(purpose: Optional[str], provider: str, model: str, seconds: float) -> None:
    usage = getattr(_last_call, "usage", None)
    _last_call.usage = None
    if not usage:
        return
    entry = {"purpose": purpose or "unlabelled", "provider": provider, "model": model,
             **usage, "seconds": round(seconds, 2)}
    log.info(
        "llm usage: purpose=%s model=%s/%s input=%d cached=%d output=%d thinking=%d seconds=%.2f",
        entry["purpose"], provider, model, entry["input_tokens"], entry["cached_tokens"],
        entry["output_tokens"], entry["thinking_tokens"], seconds,
    )
    collected = _usage_collector.get()
    if collected is not None:
        collected.append(entry)


class _Tier:
    __slots__ = ("provider", "model", "accepts_temperature", "max_attempts", "waits")

    def __init__(self, provider, model, accepts_temperature, max_attempts, waits):
        self.provider = provider
        self.model = model
        self.accepts_temperature = accepts_temperature
        self.max_attempts = max_attempts
        self.waits = waits


def _build_cascade(models: Optional[Sequence[str]] = None,
                   use_fallback: bool = True) -> List[_Tier]:
    """Assemble the tier list from defaults + env overrides (cheap; called per request).

    *models* replaces the Gemini part of the cascade for one call (e.g. a cheap
    model for a bulk job); *use_fallback* False drops the Claude tier.
    """
    gem_env = os.getenv("GEMINI_MODEL_CASCADE")
    if models:
        gem_models = [m.strip() for m in models if m and m.strip()]
    else:
        gem_models = [m.strip() for m in gem_env.split(",") if m.strip()] if gem_env else _DEFAULT_GEMINI_CASCADE

    tiers: List[_Tier] = []
    for i, m in enumerate(gem_models):
        # Fail over fast: the Gemini models share Google capacity, so when one
        # 503s the others usually do too — burning many retries just delays the
        # Claude fallback and risks Cloudflare's ~100 s request limit (the call
        # runs once for intent extraction AND once for SQL generation). Keep
        # retries minimal; the wall-clock budget in llm_generate is the backstop.
        if i < 2:
            tiers.append(_Tier("gemini", m, True, 2, [2]))   # primary: 1 retry
        else:
            tiers.append(_Tier("gemini", m, True, 1, []))    # secondary: single shot

    # cross-provider last resort — only if a key is configured
    if use_fallback and os.getenv("ANTHROPIC_API_KEY"):
        model = os.getenv("ANTHROPIC_FALLBACK_MODEL", _DEFAULT_ANTHROPIC_MODEL)
        accepts_temp = not model.startswith(_NO_TEMPERATURE_PREFIXES)
        # Single attempt: the SDK call already has its own timeout; a retry here
        # would only add latency. If it fails, the caller degrades gracefully.
        tiers.append(_Tier("anthropic", model, accepts_temp, 1, []))
    return tiers


# ── provider calls ──────────────────────────────────────────────────────────────

def _gemini_is_transient(err: str) -> bool:
    e = err.lower()
    return any(s in e for s in (
        "503", "unavailable", "429", "resource_exhausted", "rate limit",
        "deadline", "timeout", "500", "internal", "overloaded",
    ))


def _call_gemini(model, prompt, system, temperature, api_key, *, json_schema=None,
                 thinking_level=None, call_timeout=None, max_tokens=None) -> str:
    from google import genai
    from google.genai import types
    extra = {}
    if json_schema is not None:
        extra["response_mime_type"] = "application/json"
        extra["response_json_schema"] = json_schema
    # Gemini 3+ take a thinking level; 2.x models reject it with a 400.
    if thinking_level and not model.startswith("gemini-2."):
        extra["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level.upper())
    if max_tokens:
        extra["max_output_tokens"] = max_tokens
    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
        **extra,
    )
    # Per-call timeout (ms) so a hanging/slow Gemini call can't blow the wall-clock
    # budget — during a 503 storm a single call can otherwise stall ~30s.
    if call_timeout is None:
        call_timeout = float(os.getenv("GEMINI_CALL_TIMEOUT_SECONDS", "12"))
    timeout_ms = int(call_timeout * 1000)
    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(timeout=timeout_ms))
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    _note_gemini_usage(resp)
    return (resp.text or "").strip()


def _call_anthropic(model, prompt, system, temperature, accepts_temperature, max_tokens) -> str:
    import anthropic
    # Bound the fallback call: it runs for both pipeline stages, so a slow Claude
    # response must not push the request past Cloudflare's ~100s limit.
    timeout_s = float(os.getenv("ANTHROPIC_CALL_TIMEOUT_SECONDS", "20"))
    client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)  # reads ANTHROPIC_API_KEY from env
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if accepts_temperature:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    _note_anthropic_usage(resp)
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def _anthropic_is_transient(exc: Exception) -> bool:
    import anthropic
    if isinstance(exc, (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,   # covers APITimeoutError
    )):
        return True
    overloaded = getattr(anthropic, "OverloadedError", None)
    if overloaded is not None and isinstance(exc, overloaded):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return int(getattr(exc, "status_code", 0)) >= 500
    return False


# ── public entry point ───────────────────────────────────────────────────────────

def llm_generate(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    json_schema: Optional[dict] = None,
    thinking_level: Optional[str] = None,
    call_timeout: Optional[float] = None,
    budget_seconds: Optional[float] = None,
    use_fallback: bool = True,
    purpose: Optional[str] = None,
) -> str:
    """Run the prompt through the cascade; return the first tier's text response.

    The keyword options default to the pipeline's behaviour; other callers
    (README extraction, re-ranking) use them to pick their own models:
      models          Gemini models for this call, in order (replaces the cascade)
      json_schema     JSON Schema the answer must follow (Gemini structured
                      output; the Claude tier gets it as an instruction)
      thinking_level  "minimal" / "low" / "medium" / "high" on Gemini 3+
      call_timeout    seconds per Gemini call
      budget_seconds  wall-clock budget for all Gemini tiers
      use_fallback    False skips the Claude tier
      purpose         label for the usage log ("intent", "sql", ...)

    Raises LLMAllFailedError if every tier fails.
    """
    gem_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    failures: List[str] = []

    # Wall-clock budget for the Gemini tiers. Once exceeded, skip any remaining
    # Gemini models and go straight to the cross-provider (Claude) fallback, so a
    # Gemini-overload storm can't drag a single stage toward Cloudflare's limit.
    gemini_budget = (budget_seconds if budget_seconds is not None
                     else float(os.getenv("GEMINI_BUDGET_SECONDS", "12")))
    t0 = time.monotonic()
    claude_prompt = prompt
    if json_schema is not None:
        claude_prompt = (
            f"{prompt}\n\nReturn only a JSON value that follows this JSON Schema, "
            f"no prose, no code fences:\n{json.dumps(json_schema)}"
        )

    cascade = list(_build_cascade(models, use_fallback))
    gemini_tiers = [t for t in cascade if t.provider == "gemini"]
    # The last Gemini tier (the fastest model) always gets one short attempt,
    # even after the budget is spent: one slow 504 from the first model must
    # not leave the request without an answer. 0 switches this off.
    reserve = float(os.getenv("GEMINI_RESERVE_SECONDS", "12"))
    for tier in cascade:
        call_timeout_now = call_timeout
        attempts = tier.max_attempts
        if tier.provider == "gemini" and (time.monotonic() - t0) > gemini_budget:
            if reserve > 0 and gemini_tiers and tier is gemini_tiers[-1] and len(gemini_tiers) > 1:
                call_timeout_now = reserve
                attempts = 1
                log.warning("llm_generate: budget exhausted — one %.0fs reserve attempt on gemini/%s",
                            reserve, tier.model)
            else:
                failures.append(f"gemini/{tier.model}: skipped (Gemini time budget exceeded)")
                log.warning("llm_generate: skipping gemini/%s — budget exhausted, failing over",
                            tier.model)
                continue
        for attempt in range(1, attempts + 1):
            try:
                _last_call.usage = None
                call_started = time.monotonic()
                if tier.provider == "gemini":
                    if not gem_key:
                        raise RuntimeError("no GEMINI_API_KEY/GOOGLE_API_KEY configured")
                    text = _call_gemini(
                        tier.model, prompt, system, temperature, gem_key,
                        json_schema=json_schema, thinking_level=thinking_level,
                        call_timeout=call_timeout_now,
                        max_tokens=max_tokens if json_schema is not None else None,
                    )
                else:  # anthropic
                    text = _call_anthropic(
                        tier.model, claude_prompt, system, temperature,
                        tier.accepts_temperature, max_tokens,
                    )
                if text:
                    log.info("llm_generate: %s/%s succeeded", tier.provider, tier.model)
                    _report_usage(purpose, tier.provider, tier.model,
                                  time.monotonic() - call_started)
                    return text
                raise RuntimeError("empty response")
            except Exception as exc:  # noqa: BLE001 — classify below
                transient = (
                    _gemini_is_transient(str(exc)) if tier.provider == "gemini"
                    else _anthropic_is_transient(exc)
                )
                if transient and attempt < attempts:
                    time.sleep(tier.waits[attempt - 1])
                    continue
                failures.append(f"{tier.provider}/{tier.model}: {exc}")
                log.warning("llm_generate: %s/%s failed (%s)", tier.provider, tier.model,
                            "transient, tiers exhausted" if transient else "non-transient")
                break  # next tier

    raise LLMAllFailedError(
        "All LLM tiers failed:\n" + "\n".join(f"  • {f}" for f in failures)
    )
