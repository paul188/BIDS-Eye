"""
security/settings.py
--------------------
The sign-in settings, checked: what config.settings.Settings.sign_in()
returns. security/middleware.py checks them once at startup and keeps them in
app.state.auth_settings. A missing or contradictory value stops the app from
starting.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class AuthSettings(BaseModel):
    """Checked once, when security/middleware.py builds the middleware.

    A missing or contradictory value stops the app from starting, instead of
    showing up at the first sign-in or, worse, not at all.
    """

    model_config = ConfigDict(str_strip_whitespace=True, hide_input_in_errors=True)

    # Signs the session and CSRF cookies. With a missing or guessable key anyone
    # who reads the repo could sign a session for any user, so the app refuses
    # to start without one of at least 32 characters (.env.example's
    # placeholder is shorter). Generate one with
    #   python3 -c "import secrets; print(secrets.token_urlsafe(64))"
    SESSION_SECRET: SecretStr = Field(min_length=32)

    # The GitHub OAuth App the sign-in goes through.
    GITHUB_CLIENT_ID: str = Field(min_length=1)
    GITHUB_CLIENT_SECRET: SecretStr = Field(min_length=1)
    # The callback URL registered in the OAuth App. Set it in production: behind
    # a proxy that ends TLS or rewrites Host, the URL built from the request
    # can come out as http:// or on the wrong host.
    GITHUB_REDIRECT_URI: str = ""

    # Who may sign in: members of GITHUB_ORG, or every GitHub account. The
    # second has to be asked for by name, so that a lost or misspelt GITHUB_ORG
    # stops the start instead of quietly opening the site to everyone.
    GITHUB_ORG: str = ""
    ALLOW_ANY_GITHUB_USER: bool = False

    @field_validator("GITHUB_CLIENT_SECRET")
    @classmethod
    def _secret_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _org_or_explicit_open(self) -> AuthSettings:
        if not self.GITHUB_ORG and not self.ALLOW_ANY_GITHUB_USER:
            raise ValueError(
                "Set GITHUB_ORG, or ALLOW_ANY_GITHUB_USER=true to let every GitHub account sign in"
            )
        if self.GITHUB_ORG and self.ALLOW_ANY_GITHUB_USER:
            raise ValueError(
                "GITHUB_ORG and ALLOW_ANY_GITHUB_USER=true contradict each other; set only one"
            )
        return self
