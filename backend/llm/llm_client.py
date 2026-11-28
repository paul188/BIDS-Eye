"""
backend/llm/llm_client.py
------------------------------
One shared LLM call path for the whole backend.

    llm_generate(prompt, profile="sql", system=..., json_schema=...)

A *profile* (one per use case) names the tiers to try in order, each with its
retries, timeout and reserved time, plus a deadline for the whole call and the
task's defaults (temperature, max_tokens, thinking level). A tier may set its
own temperature, max_tokens or thinking level where its model needs another
value than the profile's (a thinking model needs a larger output limit). Profiles, the models
they use and the providers behind them are defined in llm_config.yaml next to
this file (or the file LLM_CONFIG_PATH names); the file names the env vars that
hold the keys, never the keys themselves.

Three layers, three places:
  * models     what a model can do (accepts temperature? takes a seed? enforces
               a JSON Schema natively? which thinking setting?): llm_config.yaml
  * profiles   policy (fallback order, retries, timeouts, deadline): llm_config.yaml
  * providers  where a model runs and how it is called: an entry in
               llm_config.yaml names its adapter `type`; one adapter class per
               type below holds the mechanics (requests, error types, usage
               counts, the API's quirks). The loop in llm_generate only sees
               the `Provider` interface and never asks which provider it talks to.

Adapter types: `gemini` (google-genai), `anthropic` (anthropic SDK) and
`openai_compatible`: any server with the OpenAI chat-completions API, which
covers local models (Ollama, vLLM, llama.cpp server, LM Studio) and hosted
ones (OpenAI, Mistral, ...). Switching a use case to another model, provider
or local hardware is a change to llm_config.yaml only.

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
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Type

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
    api_key_env: Tuple[str, ...]   # env vars that may hold the key; empty: none needed

    def available(self) -> bool:
        """Can it be called: key configured (if it needs one), SDK installed?"""

    def errors(self) -> Tuple[Type[Exception], ...]:
        """What a call raises when the provider, not the code, fails."""

    def is_transient(self, exc: Exception) -> bool:
        """Is a retry on the same model worth it?"""

    def generate(self, model: ModelSpec, req: LLMRequest, timeout: float) -> LLMResult:
        ...


def _count(obj, name: str) -> int:
    value = getattr(obj, name, None)
    return value if isinstance(value, int) else 0


def _installed(module: str) -> bool:
    """Is the SDK an adapter needs installed? (The crawler image has google-genai only.)"""
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except ImportError:  # the parent package is missing
        return False


def _first_env(names: Sequence[str]) -> Optional[str]:
    return next((os.environ[n] for n in names if os.getenv(n)), None)


def _schema_instruction(prompt: str, schema: dict) -> str:
    return (f"{prompt}\n\nReturn only a JSON value that follows this JSON Schema, "
            f"no prose, no code fences:\n{json.dumps(schema)}")


# An answer wrapped in a Markdown code fence (```json ... ```), the closing fence optional.
_CODE_FENCE_RE = re.compile(r"\s*```[\w-]*[ \t]*\n?(.*?)\s*(?:```\s*)?", re.DOTALL)


def _without_code_fence(text: str) -> str:
    """*text* without a code fence around it: models add one despite the instruction."""
    fenced = _CODE_FENCE_RE.fullmatch(text)
    return fenced.group(1) if fenced else text


# ── Gemini ──────────────────────────────────────────────────────────────────────

def _without_defaults(node: Any) -> Any:
    """The schema without `default` keywords: Gemini's responseJsonSchema accepts
    a subset of JSON Schema that leaves them out (Pydantic emits them)."""
    if isinstance(node, dict):
        return {k: _without_defaults(v) for k, v in node.items() if k != "default"}
    if isinstance(node, list):
        return [_without_defaults(v) for v in node]
    return node


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key_env: Sequence[str], seed: Optional[int] = None):
        self.api_key_env = tuple(api_key_env)
        self.seed = seed

    def available(self) -> bool:
        return _first_env(self.api_key_env) is not None and _installed("google.genai")

    def errors(self) -> Tuple[Type[Exception], ...]:
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
                extra["response_json_schema"] = _without_defaults(req.json_schema)
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
        return _first_env(self.api_key_env) is not None and _installed("anthropic")

    def errors(self) -> Tuple[Type[Exception], ...]:
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


# ── OpenAI-compatible servers (local models, OpenAI, Mistral, ...) ────────────

class OpenAICompatibleError(Exception):
    """The server answered with an error status."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


class OpenAICompatibleProvider:
    """POST {base_url}/chat/completions. A local server usually needs no key."""

    def __init__(self, name: str, base_url: str, api_key_env: Sequence[str] = (),
                 seed: Optional[int] = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = tuple(api_key_env)
        self.seed = seed

    def available(self) -> bool:
        has_key = not self.api_key_env or _first_env(self.api_key_env) is not None
        return has_key and _installed("httpx")

    def errors(self) -> Tuple[Type[Exception], ...]:
        import httpx
        return (OpenAICompatibleError, httpx.TransportError)

    def is_transient(self, exc: Exception) -> bool:
        import httpx
        if isinstance(exc, OpenAICompatibleError):
            return exc.status in (408, 429) or exc.status >= 500
        return isinstance(exc, httpx.TransportError)

    def generate(self, model: ModelSpec, req: LLMRequest, timeout: float) -> LLMResult:
        import httpx
        prompt = req.prompt
        body: dict[str, Any] = {"model": model.name, "max_tokens": req.max_tokens}
        if req.json_schema is not None:
            if model.json_schema == "native":
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "answer", "schema": req.json_schema}}
            else:
                prompt = _schema_instruction(prompt, req.json_schema)
        messages = [{"role": "system", "content": req.system}] if req.system else []
        body["messages"] = messages + [{"role": "user", "content": prompt}]
        if model.temperature:
            body["temperature"] = req.temperature
        if model.seed and self.seed is not None:
            body["seed"] = self.seed
        if req.thinking_level and model.thinking == "level":
            body["reasoning_effort"] = req.thinking_level
        headers = {}
        key = _first_env(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                          timeout=timeout)
        if resp.status_code >= 400:
            raise OpenAICompatibleError(resp.status_code, resp.text)
        try:
            data = resp.json()
            text = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleError(resp.status_code, f"unreadable answer: {exc}") from exc
        return LLMResult(text.strip(), self._usage(data.get("usage")))

    @staticmethod
    def _usage(usage) -> Optional[Dict[str, int]]:
        if not isinstance(usage, dict):
            return None
        details = usage.get("prompt_tokens_details") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        as_int = lambda v: v if isinstance(v, int) else 0  # noqa: E731
        return {
            "input_tokens": as_int(usage.get("prompt_tokens")),
            "cached_tokens": as_int(details.get("cached_tokens")),
            "output_tokens": as_int(usage.get("completion_tokens")),
            "thinking_tokens": as_int(reasoning),
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
    # This model's own values; None: the profile's.
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    thinking_level: Optional[str] = None


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


_PROVIDER_TYPES = ("gemini", "anthropic", "openai_compatible")


def _provider_from(name: str, data: dict) -> Provider:
    """One `providers` entry; `type` names its adapter (default: the entry's name)."""
    where = f"provider {name}"
    if not isinstance(data, dict):
        raise LLMConfigError(f"{where}: expected a mapping, got {data!r}")
    kind = data.get("type", name)
    key_env = data.get("api_key_env")
    key_env = [key_env] if isinstance(key_env, str) else list(key_env or [])
    if kind == "openai_compatible":
        _only(data, {"type", "base_url", "api_key_env", "seed"}, where)
        if not data.get("base_url"):
            raise LLMConfigError(f"{where}: base_url is missing")
        provider: Provider = OpenAICompatibleProvider(name, data["base_url"], key_env, data.get("seed"))
        return provider
    if kind not in _PROVIDER_TYPES:
        raise LLMConfigError(f"{where}: unknown type {kind!r} (known: {', '.join(_PROVIDER_TYPES)})")
    if not key_env:
        raise LLMConfigError(f"{where}: api_key_env is missing")
    if kind == "gemini":
        _only(data, {"type", "api_key_env", "seed"}, where)
        provider = GeminiProvider(key_env, data.get("seed"))
    else:
        _only(data, {"type", "api_key_env"}, where)
        provider = AnthropicProvider(key_env)
    provider.name = name
    return provider


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
            _only(t, {"model", "attempts", "waits", "timeout", "optional", "reserve_seconds",
                      "temperature", "max_tokens", "thinking_level"}, where)
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


def profile_available(profile: str) -> bool:
    """Can *profile* be asked at all (does any of its tiers' providers have a key)?"""
    cfg = config()
    prof = cfg.profiles.get(profile)
    if prof is None:
        raise LLMConfigError(f"LLM profile {profile!r} is not defined in {cfg.source}")
    return any(cfg.providers[cfg.models[t.model].provider].available() for t in prof.tiers)


def check_config() -> None:
    """Load the config (raising on a bad one) and warn about tiers that have no key."""
    cfg = config()
    for profile in cfg.profiles.values():
        for tier in profile.tiers:
            provider = cfg.providers[cfg.models[tier.model].provider]
            if not tier.optional and not provider.available():
                log.warning("LLM profile %s: %s cannot be called (no %s key in %s, or its SDK "
                            "is not installed); the tier will be skipped", profile.name, tier.model,
                            provider.name, " or ".join(provider.api_key_env) or "-")


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
        # Only the policy carries over; the replaced model's own temperature,
        # max_tokens and thinking level do not.
        policies = [replace(policies[min(i, len(policies) - 1)], model=name, temperature=None,
                            max_tokens=None, thinking_level=None)
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

def _first_set(*values):
    return next((v for v in values if v is not None), None)


def _request_for(tier: _Tier, prof: Profile, prompt: str, system: Optional[str],
                 json_schema: Optional[dict], thinking_level: Optional[str]) -> LLMRequest:
    """The request one tier sends: its own settings, else the profile's."""
    policy = tier.policy
    return LLMRequest(
        prompt, system,
        _first_set(policy.temperature, prof.temperature),
        _first_set(policy.max_tokens, prof.max_tokens),
        json_schema,
        _first_set(thinking_level, policy.thinking_level, prof.thinking_level),
    )


def llm_generate(
    prompt: str,
    *,
    profile: str,
    system: Optional[str] = None,
    json_schema: Optional[dict] = None,
    purpose: Optional[str] = None,
    thinking_level: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
) -> str:
    """Run the prompt through the profile's tiers; return the first text answer.

      profile         a profile of llm_config.yaml ("intent", "sql", "rerank", ...)
      json_schema     JSON Schema the answer must follow; a code fence the model
                      puts around its answer anyway is taken off
      purpose         label for the usage log; the profile's name when not given
      thinking_level  for this call, instead of the tier's or profile's (a
                      job's --thinking option)
      models          other models (defined in llm_config.yaml), in order, under
                      the profile's policy: for a bulk job's --model option

    Temperature and max_tokens come from the config only: the tier's value,
    else the profile's.

    Raises LLMAllFailedError if every tier fails, LLMConfigError for an
    unknown profile or model.
    """
    cfg = config()
    prof = cfg.profiles.get(profile)
    if prof is None:
        raise LLMConfigError(f"LLM profile {profile!r} is not defined in {cfg.source}")
    failures: List[str] = []

    tiers = []
    for tier in _tiers_for(cfg, prof, models):
        if tier.provider.available():
            tiers.append(tier)
        elif not tier.policy.optional:
            failures.append(f"{tier.label}: not available (no {tier.provider.name} key, "
                            "or its SDK is not installed)")
    if not tiers:
        raise LLMAllFailedError(f"No LLM tier of profile {profile!r} is configured:\n"
                                + "\n".join(f"  • {f}" for f in failures))

    deadline = _Deadline(prof.deadline_seconds, tiers)
    for tier in tiers:
        req = _request_for(tier, prof, prompt, system, json_schema, thinking_level)
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
            if json_schema is not None:
                return _without_code_fence(result.text)
            return result.text

    raise LLMAllFailedError(
        "All LLM tiers failed:\n" + "\n".join(f"  • {f}" for f in failures)
    )
