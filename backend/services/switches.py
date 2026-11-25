"""
services/switches.py
--------------------
The environment variables that turn a search feature on or off. Each is
"true" or "false"; unset is off. Any other value stops the backend at startup
(check, called by main.py), so a typo does not silently mean off.
"""

from __future__ import annotations

import os
from enum import Enum

from exceptions.expected import ExpectedError


class InvalidSwitchError(ExpectedError):
    """A switch is set to something other than true or false.

    Raised at startup and not caught: the backend does not start until the
    value is corrected.
    """


class Switch(str, Enum):
    """A feature switch; the value is the name of its environment variable."""
    # Text matches also search the text of the papers a dataset names.
    TEXT_SEARCH_PAPERS = "TEXT_SEARCH_PAPERS"
    # Words whose form is decided must match in that form.
    TEXT_SEARCH_EXACT = "TEXT_SEARCH_EXACT"
    # Keep datasets a filter cannot check, after the confirmed results (uncertain/tier.py).
    UNCERTAIN_TIER = "UNCERTAIN_TIER"
    # Keep datasets that do not record the field and give no hint (uncertain/conditions.py).
    UNCERTAIN_POSSIBLE = "UNCERTAIN_POSSIBLE"
    # Name each tsquery once instead of repeating its literal (uncertain/tier.py).
    UNCERTAIN_HOIST_TERMS = "UNCERTAIN_HOIST_TERMS"
    # Let a language model reorder the first results (results/rerank.py).
    RERANK_ENABLED = "RERANK_ENABLED"
    # Answer a repeated question with its stored translation (stores/interpretations.py).
    INTERPRETATION_CACHE = "INTERPRETATION_CACHE"

    def __str__(self) -> str:
        # The value, on every Python version (format() of a str enum changed in 3.11).
        return self.value

    def is_on(self) -> bool:
        """True when the switch is on. Read on every call, so a one-off
        container or a test can set the variable after import."""
        value = os.getenv(self.value, "false")
        if value not in ("true", "false"):
            raise InvalidSwitchError(f"{self.value} must be true or false, not {value!r}.")
        return value == "true"


def check() -> None:
    """Raise InvalidSwitchError when a switch has a value other than true or false."""
    for switch in Switch:
        switch.is_on()
