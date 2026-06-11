"""
LLM_preprocessor/llm_client.py
------------------------------
One shared LLM call path for the whole text-to-SQL pipeline, replacing the three
divergent Gemini cascades in preprocess.py and services/text_to_sql.py.

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
    degrade gracefully (see services/text_to_sql.py).

Config (env):
  GEMINI_API_KEY / GOOGLE_API_KEY  — Gemini key (also accepted as a call arg)
  ANTHROPIC_API_KEY                — Claude key; if absent, the Claude tier is skipped
  GEMINI_MODEL_CASCADE             — comma-separated override of the Gemini tier order
  ANTHROPIC_FALLBACK_MODEL         — override the Claude model (default claude-sonnet-4-6)
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

log = logging.getLogger(__name__)

# Claude models that reject the `temperature` sampling param (HTTP 400).
_NO_TEMPERATURE_PREFIXES = ("claude-opus-4-8", "claude-opus-4-7", "claude-fable-5")

_DEFAULT_GEMINI_CASCADE = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


class LLMAllFailedError(RuntimeError):
    """Raised when every tier in the cascade fails."""


class _Tier:
    __slots__ = ("provider", "model", "accepts_temperature", "max_attempts", "waits")

    def __init__(self, provider, model, accepts_temperature, max_attempts, waits):
        self.provider = provider
        self.model = model
        self.accepts_temperature = accepts_temperature
        self.max_attempts = max_attempts
        self.waits = waits


def _build_cascade() -> List[_Tier]:
    """Assemble the tier list from defaults + env overrides (cheap; called per request)."""
    gem_env = os.getenv("GEMINI_MODEL_CASCADE")
    gem_models = [m.strip() for m in gem_env.split(",") if m.strip()] if gem_env else _DEFAULT_GEMINI_CASCADE

    tiers: List[_Tier] = []
    for i, m in enumerate(gem_models):
        # primary two tiers get an extra retry; the rest fail over faster
        if i < 2:
            tiers.append(_Tier("gemini", m, True, 3, [2, 4]))
        else:
            tiers.append(_Tier("gemini", m, True, 2, [2]))

    # cross-provider last resort — only if a key is configured
    if os.getenv("ANTHROPIC_API_KEY"):
        model = os.getenv("ANTHROPIC_FALLBACK_MODEL", _DEFAULT_ANTHROPIC_MODEL)
        accepts_temp = not model.startswith(_NO_TEMPERATURE_PREFIXES)
        tiers.append(_Tier("anthropic", model, accepts_temp, 2, [2]))
    return tiers


# ── provider calls ──────────────────────────────────────────────────────────────

def _gemini_is_transient(err: str) -> bool:
    e = err.lower()
    return any(s in e for s in (
        "503", "unavailable", "429", "resource_exhausted", "rate limit",
        "deadline", "timeout", "500", "internal", "overloaded",
    ))


def _call_gemini(model, prompt, system, temperature, api_key) -> str:
    from google import genai
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
    )
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    return (resp.text or "").strip()


def _call_anthropic(model, prompt, system, temperature, accepts_temperature, max_tokens) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    if accepts_temperature:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
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
) -> str:
    """Run the prompt through the cascade; return the first tier's text response.

    Raises LLMAllFailedError if every tier fails.
    """
    gem_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    failures: List[str] = []

    for tier in _build_cascade():
        for attempt in range(1, tier.max_attempts + 1):
            try:
                if tier.provider == "gemini":
                    if not gem_key:
                        raise RuntimeError("no GEMINI_API_KEY/GOOGLE_API_KEY configured")
                    text = _call_gemini(
                        tier.model, prompt, system, temperature, gem_key,
                    )
                else:  # anthropic
                    text = _call_anthropic(
                        tier.model, prompt, system, temperature,
                        tier.accepts_temperature, max_tokens,
                    )
                if text:
                    log.info("llm_generate: %s/%s succeeded", tier.provider, tier.model)
                    return text
                raise RuntimeError("empty response")
            except Exception as exc:  # noqa: BLE001 — classify below
                transient = (
                    _gemini_is_transient(str(exc)) if tier.provider == "gemini"
                    else _anthropic_is_transient(exc)
                )
                if transient and attempt < tier.max_attempts:
                    time.sleep(tier.waits[attempt - 1])
                    continue
                failures.append(f"{tier.provider}/{tier.model}: {exc}")
                log.warning("llm_generate: %s/%s failed (%s)", tier.provider, tier.model,
                            "transient, tiers exhausted" if transient else "non-transient")
                break  # next tier

    raise LLMAllFailedError(
        "All LLM tiers failed:\n" + "\n".join(f"  • {f}" for f in failures)
    )
