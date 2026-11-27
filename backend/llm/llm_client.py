"""
backend/llm/llm_client.py
------------------------------
One shared LLM call path for the whole backend.

    llm_generate(prompt, profile="sql", system=..., json_schema=...)

A *profile* (one per use case) names the tiers to try in order, each with its
retries, timeout and reserved time, plus a deadline for the whole call and the
task's defaults (temperature, max_tokens, thinking level). Profiles, the models
they use and the providers behind them are defined in llm_config.yaml next to
this file (or the file LLM_CONFIG_PATH names); the file names the env vars that
hold the keys, never the keys themselves.

Three layers, three places:
  * models     what a model can do (accepts temperature? takes a seed? enforces
               a JSON Schema natively? which thinking setting?): llm_config.yaml
  * profiles   policy (fallback order, retries, timeouts, deadline): llm_config.yaml
  * providers  mechanics (SDK calls, error types, usage counts): one adapter
               class per provider below. The loop in llm_generate only sees the
               `Provider` interface and never asks which provider it talks to.

Resilience rules:
  * Retry the SAME tier (with backoff) only on TRANSIENT errors (5xx, 408,
    429, overloaded, timeouts, broken connections). On a non-transient error
    (400/401/403) skip straight to the next tier. Errors are told apart by
    type, not by words in the message. Anything else a call raises (a
    TypeError after an SDK upgrade) is a bug and is raised, instead of failing
    every tier and looking like a provider outage.
  * One deadline covers the whole call. Before every attempt the deadline
    decides how long the attempt may take, keeping back the reserve_seconds of
    every later tier, so a slow first model cannot starve the fallback.
  * A tier whose provider has no key is skipped (silently if `optional`).
  * If every tier fails, raise `LLMAllFailedError`; callers decide whether to
    degrade gracefully (see services/translate/text_to_sql.py).

The config is checked when it is first loaded (check_config() at backend
startup): an unknown model, provider or setting stops the backend instead of
failing in the middle of a request.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "llm_config.yaml"

# Read before llm_config.yaml existed; now they do nothing.
_LEGACY_ENV = ("GEMINI_MODEL_CASCADE", "ANTHROPIC_FALLBACK_MODEL", "GEMINI_SEED",
               "GEMINI_CALL_TIMEOUT_SECONDS", "ANTHROPIC_CALL_TIMEOUT_SECONDS",
               "GEMINI_BUDGET_SECONDS", "GEMINI_RESERVE_SECONDS")


class LLMAllFailedError(RuntimeError):
    """Raised when every tier in the cascade fails."""


class LLMConfigError(ValueError):
    """llm_config.yaml (or a call) names something that is not defined."""


class _EmptyResponseError(Exception):
    """A tier answered without text; the next tier is tried."""


# ── request / result ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMRequest:
    """What the caller wants, independent of the provider that answers."""
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    json_schema: Optional[dict] = None     # the answer must be JSON following this schema
    thinking_level: Optional[str] = None   # "minimal" / "low" / "medium" / "high"


@dataclass(frozen=True)
class LLMResult:
    text: str
    # input_tokens (cached ones included), cached_tokens, output_tokens,
    # thinking_tokens; None when the provider reported nothing.
    usage: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class ModelSpec:
    """What one model can do (the `models` section of llm_config.yaml)."""
    name: str
    provider: str
    temperature: bool = True       # False: the model rejects temperature, it is left out
    seed: bool = False             # True: the provider's seed is sent
    json_schema: str = "prompt"    # "native": enforced by the API; "prompt": an instruction
    thinking: str = "none"         # "level": takes a thinking level; "none": nothing is sent


class Provider(Protocol):
    name: str

    def available(self) -> bool:
        """Is a key configured?"""

    def errors(self) -> Tuple[type, ...]:
        """What a call raises when the provider, not the code, fails."""

    def is_transient(self, exc: Exception) -> bool:
        """Is a retry on the same model worth it?"""

    def generate(self, model: ModelSpec, req: LLMRequest, timeout: float) -> LLMResult:
        ...


def _count(obj, name: str) -> int:
    value = getattr(obj, name, None)
    return value if isinstance(value, int) else 0


def _first_env(names: Sequence[str]) -> Optional[str]:
    return next((os.environ[n] for n in names if os.getenv(n)), None)


def _schema_instruction(prompt: str, schema: dict) -> str:
    return (f"{prompt}\n\nReturn only a JSON value that follows this JSON Schema, "
            f"no prose, no code fences:\n{json.dumps(schema)}")


# ── Gemini ──────────────────────────────────────────────────────────────────────

class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key_env: Sequence[str], seed: Optional[int] = None):
        self.api_key_env = tuple(api_key_env)
        self.seed = seed

    def available(self) -> bool:
        return _first_env(self.api_key_env) is not None

    def errors(self) -> Tuple[type, ...]:
        # google-genai raises httpx's own errors for timeouts and broken connections.
        import httpx
        from google.genai import errors as genai_errors
        return (genai_errors.APIError, genai_errors.UnknownApiResponseError, httpx.TransportError)

    def is_transient(self, exc: Exception) -> bool:
        """5xx (UNAVAILABLE, INTERNAL, DEADLINE_EXCEEDED), 408, 429
        (RESOURCE_EXHAUSTED), a timeout or a broken connection."""
        import httpx
        from google.genai import errors as genai_errors
        if isinstance(exc, genai_errors.ServerError):
            return True
        if isinstance(exc, genai_errors.APIError):
            return exc.code in (408, 429)
        return isinstance(exc, httpx.TransportError)

    def generate(self, model: ModelSpec, req: LLMRequest, timeout: float) -> LLMResult:
        from google import genai
        from google.genai import types
        prompt = req.prompt
        extra: dict[str, Any] = {}
        if req.json_schema is not None:
            if model.json_schema == "native":
                extra["response_mime_type"] = "application/json"
                extra["response_json_schema"] = req.json_schema
            else:
                prompt = _schema_instruction(prompt, req.json_schema)
            # The output cap is set only for JSON answers: Gemini 2.5 spends
            # output tokens on thinking first, and the free-text callers (SQL)
            # rely on the model's own, larger default.
            extra["max_output_tokens"] = req.max_tokens
        if req.thinking_level and model.thinking == "level":
            extra["thinking_config"] = types.ThinkingConfig(thinking_level=req.thinking_level.upper())
        if model.seed and self.seed is not None:
            extra["seed"] = self.seed
        cfg = types.GenerateContentConfig(
            system_instruction=req.system or None,
            temperature=req.temperature if model.temperature else None,
            **extra,
        )
        client = genai.Client(api_key=_first_env(self.api_key_env),
                              http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        resp = client.models.generate_content(model=model.name, contents=prompt, config=cfg)
        return LLMResult((resp.text or "").strip(), self._usage(resp))

    @staticmethod
    def _usage(resp) -> Optional[Dict[str, int]]:
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return None
        return {
            "input_tokens": _count(meta, "prompt_token_count"),
            "cached_tokens": _count(meta, "cached_content_token_count"),
            "output_tokens": _count(meta, "candidates_token_count"),
            "thinking_tokens": _count(meta, "thoughts_token_count"),
        }


# ── Anthropic ───────────────────────────────────────────────────────────────────

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key_env: Sequence[str]):
        self.api_key_env = tuple(api_key_env)

    def available(self) -> bool:
        return _first_env(self.api_key_env) is not None

    def errors(self) -> Tuple[type, ...]:
        # anthropic wraps timeouts and broken connections in APIConnectionError.
        import anthropic
        return (anthropic.APIError,)

    def is_transient(self, exc: Exception) -> bool:
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

    def generate(self, model: ModelSpec, req: LLMRequest, timeout: float) -> LLMResult:
        import anthropic
        prompt = req.prompt
        if req.json_schema is not None:
            prompt = _schema_instruction(prompt, req.json_schema)
        # The loop does the retrying; the SDK's own retries would hide time from the deadline.
        client = anthropic.Anthropic(api_key=_first_env(self.api_key_env),
                                     timeout=timeout, max_retries=0)
        kwargs: dict[str, Any] = {
            "model": model.name,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if req.system:
            kwargs["system"] = req.system
        if model.temperature:
            kwargs["temperature"] = req.temperature
        resp = client.messages.create(**kwargs)
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
        return LLMResult(text, self._usage(resp))

    @staticmethod
    def _usage(resp) -> Optional[Dict[str, int]]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        cached = _count(usage, "cache_read_input_tokens")
        return {
            # Anthropic reports uncached, cache-read and cache-write input apart.
            "input_tokens": (_count(usage, "input_tokens") + cached
                             + _count(usage, "cache_creation_input_tokens")),
            "cached_tokens": cached,
            "output_tokens": _count(usage, "output_tokens"),
            "thinking_tokens": 0,
        }


# ── config ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TierConfig:
    model: str
    attempts: int = 1
    waits: Tuple[float, ...] = ()     # seconds before retry 1, 2, ...
    timeout: float = 12.0             # seconds per attempt
    optional: bool = False            # no key: skipped without a failure entry
    reserve_seconds: float = 0.0      # kept back for this tier while earlier tiers run


@dataclass(frozen=True)
class Profile:
    name: str
    tiers: Tuple[TierConfig, ...]
    deadline_seconds: float
    temperature: float = 0.0
    max_tokens: int = 2048
    thinking_level: Optional[str] = None


@dataclass(frozen=True)
class LLMConfig:
    models: Dict[str, ModelSpec]
    providers: Dict[str, Provider]
    profiles: Dict[str, Profile]
    source: str = "<dict>"


def _provider_from(name: str, data: dict) -> Provider:
    key_env = data.get("api_key_env")
    key_env = [key_env] if isinstance(key_env, str) else list(key_env or [])
    if not key_env:
        raise LLMConfigError(f"provider {name}: api_key_env is missing")
    if name == "gemini":
        _only(data, {"api_key_env", "seed"}, f"provider {name}")
        return GeminiProvider(key_env, data.get("seed"))
    if name == "anthropic":
        _only(data, {"api_key_env"}, f"provider {name}")
        return AnthropicProvider(key_env)
    raise LLMConfigError(f"unknown provider {name!r} (known: gemini, anthropic)")


def _only(data: dict, allowed: set, where: str) -> None:
    if not isinstance(data, dict):
        raise LLMConfigError(f"{where}: expected a mapping, got {data!r}")
    unknown = set(data) - allowed
    if unknown:
        raise LLMConfigError(f"{where}: unknown setting(s) {sorted(unknown)}")


def parse_config(data: dict, source: str = "<dict>") -> LLMConfig:
    """Check the config and build it; raise LLMConfigError on the first problem."""
    _only(data, {"providers", "models", "profiles"}, source)
    providers = {name: _provider_from(name, d) for name, d in (data.get("providers") or {}).items()}

    models: Dict[str, ModelSpec] = {}
    for name, d in (data.get("models") or {}).items():
        _only(d, {"provider", "temperature", "seed", "json_schema", "thinking"}, f"model {name}")
        spec = ModelSpec(name=name, **d)
        if spec.provider not in providers:
            raise LLMConfigError(f"model {name}: provider {spec.provider!r} is not defined")
        if spec.json_schema not in ("native", "prompt"):
            raise LLMConfigError(f"model {name}: json_schema must be native or prompt")
        if spec.thinking not in ("level", "none"):
            raise LLMConfigError(f"model {name}: thinking must be level or none")
        models[name] = spec

    profiles: Dict[str, Profile] = {}
    for name, d in (data.get("profiles") or {}).items():
        _only(d, {"tiers", "deadline_seconds", "temperature", "max_tokens", "thinking_level"},
              f"profile {name}")
        tiers = []
        for i, t in enumerate(d.get("tiers") or []):
            where = f"profile {name}, tier {i + 1}"
            _only(t, {"model", "attempts", "waits", "timeout", "optional", "reserve_seconds"}, where)
            tier = TierConfig(**{**t, "waits": tuple(t.get("waits", ()))})
            if tier.model not in models:
                raise LLMConfigError(f"{where}: model {tier.model!r} is not defined")
            if tier.attempts < 1 or len(tier.waits) < tier.attempts - 1:
                raise LLMConfigError(f"{where}: needs attempts >= 1 and a wait before each retry")
            if tier.timeout <= 0 or tier.reserve_seconds < 0:
                raise LLMConfigError(f"{where}: timeout must be > 0, reserve_seconds >= 0")
            tiers.append(tier)
        if not tiers:
            raise LLMConfigError(f"profile {name}: no tiers")
        if "deadline_seconds" not in d:
            raise LLMConfigError(f"profile {name}: deadline_seconds is missing")
        profiles[name] = Profile(name=name, tiers=tuple(tiers),
                                 **{k: v for k, v in d.items() if k != "tiers"})
    return LLMConfig(models, providers, profiles, source)


def config_path() -> Path:
    """The config file in use: LLM_CONFIG_PATH, else llm_config.yaml next to this file."""
    return Path(os.getenv("LLM_CONFIG_PATH") or DEFAULT_CONFIG_PATH)


def load_config(path: Optional[Path] = None) -> LLMConfig:
    import yaml
    path = path or config_path()
    with open(path, encoding="utf-8") as fh:
        return parse_config(yaml.safe_load(fh) or {}, str(path))


_config: Optional[LLMConfig] = None


def config() -> LLMConfig:
    global _config
    if _config is None:
        _config = load_config()
        stale = [n for n in _LEGACY_ENV if os.getenv(n)]
        if stale:
            log.warning("llm_client: %s set but ignored; models, seed and timeouts are in %s",
                        ", ".join(stale), _config.source)
    return _config


def check_config() -> None:
    """Load the config (raising on a bad one) and warn about tiers that have no key."""
    cfg = config()
    for profile in cfg.profiles.values():
        for tier in profile.tiers:
            provider = cfg.providers[cfg.models[tier.model].provider]
            if not tier.optional and not provider.available():
                log.warning("LLM profile %s: %s has no %s key (%s); the tier will be skipped",
                            profile.name, tier.model, provider.name, " or ".join(provider.api_key_env))


# ── tiers and deadline ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Tier:
    provider: Provider
    model: ModelSpec
    policy: TierConfig

    @property
    def label(self) -> str:
        return f"{self.provider.name}/{self.model.name}"


def _tiers_for(cfg: LLMConfig, profile: Profile, models: Optional[Sequence[str]]) -> List[_Tier]:
    """The profile's tiers; with *models*, those models under the profile's policy.

    Model i takes the retries, timeout and reserve of the profile's tier i (the
    last tier's for any beyond), so a bulk job can pick its model on the command
    line without restating the policy.
    """
    policies = list(profile.tiers)
    if models:
        names = [m.strip() for m in models if m and m.strip()]
        policies = [replace(policies[min(i, len(policies) - 1)], model=name)
                    for i, name in enumerate(names)]
    tiers = []
    for policy in policies:
        spec = cfg.models.get(policy.model)
        if spec is None:
            raise LLMConfigError(f"model {policy.model!r} is not defined in {cfg.source}")
        tiers.append(_Tier(cfg.providers[spec.provider], spec, policy))
    return tiers


# An attempt shorter than this cannot answer a real prompt; the tier is skipped.
_MIN_ATTEMPT_SECONDS = 2.0


class _Deadline:
    """One time budget over all tiers of a call; each tier keeps the
    reserve_seconds of every later tier free."""

    def __init__(self, seconds: float, tiers: List[_Tier]):
        self._end = time.monotonic() + seconds
        self._reserved_after = {
            id(t): sum(later.policy.reserve_seconds for later in tiers[i + 1:])
            for i, t in enumerate(tiers)
        }

    def timeout_for(self, tier: _Tier) -> Optional[float]:
        """Seconds the next attempt may take; None when there is no time left."""
        left = self._end - time.monotonic() - self._reserved_after[id(tier)]
        seconds = min(tier.policy.timeout, left)
        return seconds if seconds >= _MIN_ATTEMPT_SECONDS else None


# ── token usage ─────────────────────────────────────────────────────────────────
# Every successful call is logged with its token counts, and added to the list
# started by start_usage_collection() when there is one, so the cost of one
# search can be summed per stage. Failed attempts report nothing.

_usage_collector: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "llm_usage_collector", default=None)


def start_usage_collection() -> list:
    """Collect the usage of every later call in this context; returns the list.

    Each entry: purpose, provider, model, input_tokens (cached ones included),
    cached_tokens, output_tokens, thinking_tokens, seconds.
    """
    collected: list = []
    _usage_collector.set(collected)
    return collected


def _report_usage(purpose: str, tier: _Tier, usage: Optional[Dict[str, int]],
                  seconds: float) -> None:
    if not usage:
        return
    entry = {"purpose": purpose, "provider": tier.provider.name, "model": tier.model.name,
             **usage, "seconds": round(seconds, 2)}
    log.info(
        "llm usage: purpose=%s model=%s input=%d cached=%d output=%d thinking=%d seconds=%.2f",
        purpose, tier.label, entry["input_tokens"], entry["cached_tokens"],
        entry["output_tokens"], entry["thinking_tokens"], seconds,
    )
    collected = _usage_collector.get()
    if collected is not None:
        collected.append(entry)


# ── public entry point ───────────────────────────────────────────────────────────

def llm_generate(
    prompt: str,
    *,
    profile: str,
    system: Optional[str] = None,
    json_schema: Optional[dict] = None,
    purpose: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    thinking_level: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
) -> str:
    """Run the prompt through the profile's tiers; return the first text answer.

      profile         a profile of llm_config.yaml ("intent", "sql", "rerank", ...)
      json_schema     JSON Schema the answer must follow
      purpose         label for the usage log; the profile's name when not given
      temperature, max_tokens, thinking_level
                      override the profile's defaults for this call
      models          other models (defined in llm_config.yaml), in order, under
                      the profile's policy: for a bulk job's --model option

    Raises LLMAllFailedError if every tier fails, LLMConfigError for an
    unknown profile or model.
    """
    cfg = config()
    prof = cfg.profiles.get(profile)
    if prof is None:
        raise LLMConfigError(f"LLM profile {profile!r} is not defined in {cfg.source}")
    req = LLMRequest(
        prompt, system,
        prof.temperature if temperature is None else temperature,
        prof.max_tokens if max_tokens is None else max_tokens,
        json_schema,
        prof.thinking_level if thinking_level is None else thinking_level,
    )
    failures: List[str] = []

    tiers = []
    for tier in _tiers_for(cfg, prof, models):
        if tier.provider.available():
            tiers.append(tier)
        elif not tier.policy.optional:
            failures.append(f"{tier.label}: no {tier.provider.name} API key configured")
    if not tiers:
        raise LLMAllFailedError(f"No LLM tier of profile {profile!r} is configured:\n"
                                + "\n".join(f"  • {f}" for f in failures))

    deadline = _Deadline(prof.deadline_seconds, tiers)
    for tier in tiers:
        expected = tier.provider.errors() + (_EmptyResponseError,)
        for attempt in range(1, tier.policy.attempts + 1):
            timeout = deadline.timeout_for(tier)
            if timeout is None:
                failures.append(f"{tier.label}: skipped (time budget spent)")
                log.warning("llm_generate: skipping %s, time budget spent", tier.label)
                break
            try:
                started = time.monotonic()
                result = tier.provider.generate(tier.model, req, timeout)
                if not result.text:
                    raise _EmptyResponseError("empty response")
            except expected as exc:  # anything else is a bug and is raised
                transient = (not isinstance(exc, _EmptyResponseError)
                             and tier.provider.is_transient(exc))
                if transient and attempt < tier.policy.attempts:
                    time.sleep(tier.policy.waits[attempt - 1])
                    continue
                failures.append(f"{tier.label}: {exc}")
                log.warning("llm_generate: %s failed (%s)", tier.label,
                            "transient, retries used up" if transient else "non-transient")
                break  # next tier
            log.info("llm_generate: %s succeeded", tier.label)
            _report_usage(purpose or profile, tier, result.usage, time.monotonic() - started)
            return result.text

    raise LLMAllFailedError(
        "All LLM tiers failed:\n" + "\n".join(f"  • {f}" for f in failures)
    )
