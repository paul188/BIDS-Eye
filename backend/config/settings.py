"""
config/settings.py
------------------
Every environment variable the backend reads, typed, in one object.

    from config.settings import settings_obj
    settings_obj.DAILY_MESSAGE_LIMIT

The environment is read once, when this module is first imported. A value of
the wrong type stops the import, and with it the start of the backend. A test
changes a setting with monkeypatch.setattr(settings_obj, "NAME", value), or
builds its own Settings() after setting the environment.

Read elsewhere: the database connection (METADATA_DB_*) by BIDS-SQL/settings.py;
the model keys, LLM_CONFIG_PATH and TOPIC_MAX_ALTERNATIVE_MATCHES by backend/llm/;
the CRAWL_* variables by the crawler.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BeforeValidator, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from security.settings import AuthSettings


def _true_or_false(value: object) -> object:
    if isinstance(value, bool):
        return value
    if value not in ("true", "false"):
        raise ValueError("must be true or false")
    return value == "true"


# A feature switch (services/switches.py): "true" or "false", nothing else, so
# that a typo does not silently mean off.
OnOff = Annotated[bool, BeforeValidator(_true_or_false)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", hide_input_in_errors=True)

    # ── Sign-in (security/settings.py checks them) ────────────────────────────
    # Not required here: the crawler imports backend code without them.
    # sign_in() checks them when the app is built.
    SESSION_SECRET: SecretStr = SecretStr("")
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: SecretStr = SecretStr("")
    GITHUB_REDIRECT_URI: str = ""
    GITHUB_ORG: str = ""
    ALLOW_ANY_GITHUB_USER: bool = False
    # The one other origin that may call the API; unset or * means none.
    FRONTEND_URL: str = ""

    # ── Rate limits (security/rate_limits.py) ─────────────────────────────────
    # Searches per user per UTC day.
    DAILY_MESSAGE_LIMIT: int = 20
    # Comma-separated GitHub logins exempt from the daily cap.
    ADMIN_GITHUB_LOGINS: str = "paul188"

    # ── Logging (request_log.py) ──────────────────────────────────────────────
    LOG_FORMAT: Literal["text", "json"] = "text"

    # ── Language models (which model: backend/llm/llm_config.yaml) ────────────
    # Seconds the whole translation may take (services/translate/text_to_sql.py).
    PIPELINE_DEADLINE_SECONDS: float = 80
    DATASET_DESCRIPTION_MODEL: str = "gemini-2.5-flash"
    README_PROFILE_MODEL: str = "gemini-3.8-flash"
    # Empty for none.
    README_PROFILE_THINKING: Optional[str] = None

    # ── Feature switches (services/switches.py) ───────────────────────────────
    TEXT_SEARCH_PAPERS: OnOff = False
    TEXT_SEARCH_EXACT: OnOff = False
    UNCERTAIN_TIER: OnOff = False
    UNCERTAIN_POSSIBLE: OnOff = False
    UNCERTAIN_HOIST_TERMS: OnOff = False
    RERANK_ENABLED: OnOff = False
    INTERPRETATION_CACHE: OnOff = False

    # Conditions per dataset that may be unconfirmed (services/uncertain/tier.py).
    UNCERTAIN_MAX_UNSURE: int = 1
    # Drop values the catalogue does not hold (services/translate/value_check.py).
    CATALOGUE_VALUE_CHECK: bool = True

    # ── Rerank (services/results/rerank.py) ───────────────────────────────────
    # Comma-separated, tried in order; empty: the profile in llm_config.yaml.
    RERANK_MODEL: str = ""
    # Empty for none.
    RERANK_THINKING: Optional[str] = None
    RERANK_CANDIDATES: int = 40
    RERANK_BATCH: int = 20
    RERANK_TIMEOUT_SECONDS: float = 20

    # ── Limits on generated SQL (services/query_guard.py) ─────────────────────
    # Unset: 3 × the largest table's row estimate.
    SEARCH_MAX_PLAN_ROWS: Optional[float] = None
    SEARCH_STATEMENT_TIMEOUT: str = "120s"
    SEARCH_WORK_MEM: str = "32MB"
    SEARCH_TEMP_FILE_LIMIT: str = "1GB"
    SEARCH_RANKED_PAGE_TIMEOUT: str = "25s"
    SEARCH_MAX_CONCURRENT: int = 2
    SEARCH_SLOT_WAIT_SECONDS: float = 30

    # ── Other ─────────────────────────────────────────────────────────────────
    # The name index's database back-check (services/translate/text_to_sql.py).
    DATABASE_URL: Optional[str] = None
    # Run the crawler in this process (services/in_process_crawler.py).
    RUN_CRAWLER_IN_PROCESS: bool = False

    @field_validator("README_PROFILE_THINKING", "RERANK_THINKING", "SEARCH_MAX_PLAN_ROWS",
                     "DATABASE_URL", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def admin_logins(self) -> frozenset[str]:
        return frozenset(u.strip() for u in self.ADMIN_GITHUB_LOGINS.split(",") if u.strip())

    def sign_in(self) -> AuthSettings:
        """The sign-in settings, checked. Raises pydantic.ValidationError
        naming what is missing or contradictory."""
        return AuthSettings.model_validate(
            {name: getattr(self, name) for name in AuthSettings.model_fields}
        )


settings_obj = Settings()
