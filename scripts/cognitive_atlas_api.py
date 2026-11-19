"""
scripts/cognitive_atlas_api.py
------------------------------
Reading the Cognitive Atlas API (cognitiveatlas.org), shared by
import_cognitive_atlas.py, generate_task_bridge.py and fill_bridge_gaps.py.

get_json() tells "there is nothing at this URL" (404: None) apart from "the
API could not be read" (CognitiveAtlasFetchError, after retries). The
scripts used to get None for both, and wrote a concept without its
hierarchy, or a task without its constructs, into a file they overwrite in
full.
"""
from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional, Union

Json = Union[dict, list]


class CognitiveAtlasFetchError(Exception):
    """A Cognitive Atlas URL could not be read, after retries."""


def _retry_after(headers, attempt: int) -> int:
    try:
        return int(headers.get("Retry-After", ""))
    except ValueError:
        return 2 ** (attempt + 1)


def get_json(url: str, retries: int = 5) -> Optional[Json]:
    """The JSON at *url*, or None when it does not exist (404).

    Retries rate limits (429, honouring Retry-After), server errors (5xx),
    network errors and answers that are not JSON. Raises
    CognitiveAtlasFetchError when every attempt failed or the API refused
    the request (another 4xx)."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 429:
                wait = _retry_after(exc.headers, attempt)
                print(f"  [429] rate limited — waiting {wait}s …", file=sys.stderr)
                time.sleep(wait)
                last = exc
                continue
            if exc.code < 500:
                raise CognitiveAtlasFetchError(f"{url}: {exc}") from exc
            last = exc
        # URLError and timeouts are OSErrors; IncompleteRead is an
        # HTTPException; ValueError covers JSONDecodeError and UnicodeDecodeError.
        except (OSError, http.client.HTTPException, ValueError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise CognitiveAtlasFetchError(f"{url}: {last}") from last
