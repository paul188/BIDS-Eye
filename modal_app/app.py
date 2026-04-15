"""
modal_app/app.py
----------------
Modal serverless inference for BIDS-Eye Text-to-SQL.

Architecture
────────────
1. Vector-augmented retrieval (RAG):
   A metadata index (built by build_metadata_index.py and stored in the
   "bids-eye-metadata" Modal Volume) maps known DB values — tasks, datatypes,
   suffixes, diagnoses — to sentence embeddings.  On each request the question
   is embedded and the top-k most semantically similar values are injected as
   a context hint before the question.  This prevents the model from inventing
   field values like `task = 'working memory'` when the real value is `'nback'`.

2. LLM inference:
   Phi-3-mini-128k-instruct + QLoRA LoRA adapters, loaded from a Modal Volume.
   Prompt format is Alpaca (### Instruction / ### Question / ### SQL) — this
   MUST match the format used in training/train.py.

One-time setup:
   python modal_app/build_metadata_index.py   # on HPC (needs DB access)
   bash  modal_app/upload_index.sh            # uploads index to Modal volume
   modal deploy modal_app/app.py

Backend calls:
   TextToSQLModel.generate.remote.aio(question)
   → {"sql": "SELECT ...", "augmented_question": "...", "context_hints": {...}}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import modal

# ── Volumes ────────────────────────────────────────────────────────────────────
adapters_volume  = modal.Volume.from_name("bids-eye-weights",   create_if_missing=True)
metadata_volume  = modal.Volume.from_name("bids-eye-metadata",  create_if_missing=True)
ADAPTERS_PATH    = "/adapters"
METADATA_PATH    = "/metadata"
INDEX_FILE       = f"{METADATA_PATH}/metadata_index.json"

HF_MODEL = "microsoft/Phi-3-mini-128k-instruct"

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "einops",
        "sentencepiece",
        "huggingface_hub>=0.22.0",
        "sentence-transformers>=2.7.0",
        "numpy>=1.26.0",
    )
    # Bake constants.py into the image so prompt.py can import it
    .copy_local_file(
        "training_data_generation/constants.py",
        "/app/constants.py",
    )
)

app = modal.App("bids-eye", image=image)


# ── SQL extraction ─────────────────────────────────────────────────────────────
# The model is trained with Alpaca format and outputs raw SQL after "### SQL:\n".
# No markdown fences are expected, but we strip them if the model adds them.

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)

_FALLBACK_SQL = (
    "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
    "       d.source_type, d.remote_url, d.validation_status,\n"
    "       COUNT(DISTINCT o.subject) AS subject_count\n"
    "FROM bids_datasets d\n"
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
    "GROUP BY d.id\n"
    "ORDER BY d.name\n"
    "LIMIT 50"
)


def _extract_sql(raw: str) -> str:
    """Extract clean SQL from raw model output."""
    # Strip any accidental fence the model added
    fence_match = _SQL_FENCE.search(raw)
    if fence_match:
        return fence_match.group(1).strip()
    # Raw SQL (expected path for Alpaca-trained model)
    stripped = raw.strip()
    if re.match(r"(?i)select\b", stripped):
        # Trim at first blank line (guard against rambling)
        return stripped.split("\n\n")[0].strip()
    return _FALLBACK_SQL


# ── Vector retrieval ───────────────────────────────────────────────────────────

class MetadataRetriever:
    """
    Lightweight RAG retriever over known BIDS DB values.

    The index is a JSON file built by build_metadata_index.py:
      {
        "values": {"tasks": [...], "datatypes": [...], "suffixes": [...], "diagnoses": [...]},
        "embeddings": {"tasks": [[...], ...], "diagnoses": [[...], ...], ...}
      }

    At query time we embed the question and return the top-k values per category
    that are most semantically similar.  These are injected into the prompt as
    context hints so the model uses real DB values rather than inventing them.
    """

    CATEGORIES = ["tasks", "datatypes", "suffixes", "diagnoses"]
    TOP_K = 3

    def __init__(self, index_path: str):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)

        self._values: Dict[str, List[str]] = idx["values"]
        self._embeddings: Dict[str, np.ndarray] = {
            cat: np.array(idx["embeddings"][cat], dtype="float32")
            for cat in self.CATEGORIES
            if cat in idx.get("embeddings", {})
        }

    def retrieve(self, question: str) -> Dict[str, List[str]]:
        """Return {category: [top-k matching values]} for a question."""
        import numpy as np

        q_emb = self._model.encode([question], normalize_embeddings=True)[0]
        hints: Dict[str, List[str]] = {}

        for cat in self.CATEGORIES:
            if cat not in self._embeddings or len(self._values.get(cat, [])) == 0:
                continue
            sims = self._embeddings[cat] @ q_emb          # cosine (embeddings are L2-normalised)
            top_k = min(self.TOP_K, len(sims))
            top_idx = np.argsort(sims)[::-1][:top_k]
            top_vals = [self._values[cat][i] for i in top_idx if sims[i] > 0.25]
            if top_vals:
                hints[cat] = top_vals

        return hints


def _build_augmented_question(question: str, hints: Dict[str, List[str]]) -> str:
    """Prepend DB context hints to the question so the model uses real values."""
    if not hints:
        return question
    parts = []
    label_map = {
        "tasks": "relevant task values",
        "datatypes": "relevant datatype values",
        "suffixes": "relevant suffix values",
        "diagnoses": "relevant diagnosis values",
    }
    for cat, vals in hints.items():
        parts.append(f"{label_map.get(cat, cat)}: {', '.join(vals)}")
    context_block = "[DB context — use these exact values in your SQL if appropriate]\n" + "\n".join(parts)
    return f"{context_block}\n\n{question}"


# ── Modal class ────────────────────────────────────────────────────────────────

@app.cls(
    gpu="T4",
    volumes={
        ADAPTERS_PATH: adapters_volume,
        METADATA_PATH: metadata_volume,
    },
    timeout=120,
    scaledown_window=300,
)
class TextToSQLModel:
    """Phi-3-mini-128k-instruct + QLoRA adapters, served on Modal."""

    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        snapshot_download(HF_MODEL)
        # Also cache the sentence-transformers retrieval model
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @modal.enter()
    def load_model(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ── System prompt (Alpaca format, must match training/train.py) ──────
        from .constants import SYSTEM
        self._system = SYSTEM

        # ── Tokeniser ────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Base model + LoRA adapters ────────────────────────────────────────
        base = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, ADAPTERS_PATH)
        self.model.eval()

        # ── Metadata retriever (optional — skip if index not uploaded yet) ───
        self._retriever: Optional[MetadataRetriever] = None
        if Path(INDEX_FILE).exists():
            try:
                self._retriever = MetadataRetriever(INDEX_FILE)
                print(f"[bids-eye] Metadata retriever loaded from {INDEX_FILE}")
            except Exception as exc:
                print(f"[bids-eye] WARNING: could not load metadata index: {exc}")
        else:
            print(f"[bids-eye] No metadata index at {INDEX_FILE} — RAG disabled. "
                  "Run build_metadata_index.py to enable it.")

    @modal.method()
    def generate(self, question: str, max_new_tokens: int = 512) -> dict:
        """
        Translate a natural-language question into a SQL query.

        Returns:
          sql                — the generated SQL string
          augmented_question — the question actually fed to the model (with context hints)
          context_hints      — {category: [matched values]} from the retriever, or {}
        """
        import torch

        # ── Step 1: retrieve context hints ───────────────────────────────────
        hints: Dict[str, List[str]] = {}
        if self._retriever is not None:
            hints = self._retriever.retrieve(question)

        augmented = _build_augmented_question(question, hints)

        # ── Step 2: build Alpaca-format prompt ───────────────────────────────
        # Format MUST match training/train.py format_inference()
        prompt = (
            f"### Instruction:\n{self._system}\n\n"
            f"### Question:\n{augmented.strip()}\n\n"
            f"### SQL:\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to("cuda")

        # ── Step 3: generate ─────────────────────────────────────────────────
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        sql = _extract_sql(raw)

        return {
            "sql": sql,
            "augmented_question": augmented,
            "context_hints": hints,
        }
