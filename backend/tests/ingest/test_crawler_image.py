"""The crawler image holds the code its post-crawl steps import.

post_crawl.py only logs a step it cannot import, so a file missing from
ingest/openneuro-crawler/Dockerfile would go unnoticed. The test copies what
the Dockerfile copies into an empty folder and imports the steps there.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "ingest" / "openneuro-crawler" / "Dockerfile"

if not DOCKERFILE.is_file() or not (ROOT / "BIDS-SQL" / "db").is_dir():
    pytest.skip("the crawler's Dockerfile or BIDS-SQL is not next to this checkout (e.g. inside "
                "the image)", allow_module_level=True)

_CHECK = """
import importlib.util
import sys
import post_crawl
assert post_crawl._descriptions_available(), "dataset descriptions would be skipped"
assert post_crawl._backend_services_available(), "word forms would be skipped"
assert post_crawl._profiles_available(), "README profiles would be skipped"
from remote import participant_codes, participant_groups
participant_codes._vocabulary_file()
origins = {name: getattr(module, "__file__", None) for name, module in sys.modules.items()}
# participant_groups imports these when it runs. Found, not imported: reading
# the vocabulary takes half a minute.
for name in ("yaml_to_llamaindex", "text_terms"):
    origins[name] = importlib.util.find_spec(name).origin
outside = sorted(name for name, path in origins.items() if (path or "").startswith(sys.argv[1]))
assert not outside, f"imported from the checkout, not from the image: {outside}"
"""


def test_post_crawl_steps_import_in_the_image(tmp_path):
    pytest.importorskip("google.genai")
    app = tmp_path / "app"
    dockerfile = DOCKERFILE.read_text()
    for source, target in re.findall(r"^COPY\s+(\S+)\s+/app/(\S+)\s*$", dockerfile, re.MULTILINE):
        start, end = ROOT / source, app / target
        if start.is_dir():
            shutil.copytree(start, end, dirs_exist_ok=True, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", ".git", "*venv", "embed_*"))
        else:
            end.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(start, end)
    installed = re.findall(r"-e /app/(\S+)", dockerfile)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("COV_")}
    environment |= {"PYTHONPATH": os.pathsep.join(str(app / folder) for folder in installed),
                    "PYTHONDONTWRITEBYTECODE": "1", "GEMINI_API_KEY": "not-a-key"}
    for name in ("NAME", "USER", "PASSWORD", "HOST"):
        environment.setdefault(f"METADATA_DB_{name}", "ci")
    checked = subprocess.run([sys.executable, "-c", _CHECK, f"{ROOT}{os.sep}"], cwd=app / "crawler",
                             env=environment, capture_output=True, text=True, timeout=120)
    assert checked.returncode == 0, checked.stderr[-2000:]
