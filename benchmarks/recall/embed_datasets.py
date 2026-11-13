"""
benchmarks/recall/embed_datasets.py
-----------------------------------
Embed every dataset of a snapshot with a general-purpose retrieval model, for
the embedding source of build_pool.py.

    python benchmarks/recall/embed_datasets.py SNAPSHOT.jsonl OUT_DIR

Writes doc_emb.npy (normalised float32 vectors), doc_ids.json (accessions in
the same order) and model.json. About 15 minutes on a laptop CPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MODEL = "BAAI/bge-base-en-v1.5"


def document(r: dict) -> str:
    """The text a dataset is represented by: names, uploader fields, codes, README start."""
    parts = [
        r.get("name") or "", r.get("dx_status") or "", r.get("study_domain") or "",
        r.get("study_design") or "",
        "modalities: " + ", ".join(r.get("datatypes") or []),
        "tasks: " + ", ".join((r.get("tasks") or [])[:20]),
        (r.get("description") or "")[:600], (r.get("readme") or "")[:1800],
    ]
    return "\n".join(p for p in parts if p.strip())


def main() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    snapshot, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in snapshot.open() if l.startswith("{")]
    rows = [r for r in rows if r.get("acc")]
    model = SentenceTransformer(MODEL)
    started = time.time()
    emb = model.encode([document(r) for r in rows], batch_size=16,
                       normalize_embeddings=True, show_progress_bar=False)
    np.save(out / "doc_emb.npy", emb.astype(np.float32))
    (out / "doc_ids.json").write_text(json.dumps([r["acc"] for r in rows]))
    (out / "model.json").write_text(json.dumps({"model": MODEL}))
    print(f"embedded {len(rows)} datasets in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
