"""
modal_app/app.py
----------------
Modal serverless inference for BIDS-Eye Text-to-SQL.

End-to-end pipeline on a single call to TextToSQLModel.run():

  User question  (natural language)
       │
       ▼  1. Preprocessing  — Gemini parses question → QueryPlan (Pydantic)
       │                       Identifies query family, extracts terms per field
       │
       ▼  2. RAG             — MetadataRetriever (RAG/retriever.py) resolves:
       │                         diagnosis / task / suffix / datatype  via value_mappings.yaml
       │                         author / funding                       via name_index.json
       │                       Returns AugmentedQueryPlan with canonical DB codes injected
       │
       ▼  3. Few-shot inject — Picks 5 examples from few_shot_examples.json
       │                       matching the same QueryFamily as this request
       │
       ▼  4. SQLCoder        — defog/sqlcoder-7b-2 + QLoRA adapters generates SQL
       │                       from the augmented question + schema + examples
       │
       ▼  5. SQL expand      — sql_expander.py expands concept codes → IN (...) lists
       │
       ▼  6. DB query        — Optional: executes SQL against PostgreSQL, returns rows

One-time setup:
  python RAG/build_name_index.py --db-url <url> --out /path/to/name_index.json
  modal volume put bids-eye-metadata name_index.json /name_index.json
  modal secret create bids-eye-gemini GEMINI_API_KEY=<key>
  modal deploy modal_app/app.py

Calling:
  model = modal.Cls.from_name("bids-eye", "TextToSQLModel")
  result = model.run.remote("Find datasets with fMRI and n-back tasks")
  print(result["sql"])
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# Read deploy-time tuning knobs from the environment (set in .env before deploying).
# MODAL_MIN_CONTAINERS=1  → keep one container always warm (no cold starts, ~$1/hr for A10G)
# MODAL_MAX_NEW_TOKENS    → token budget for SQLCoder; 300 is plenty for SQL, 512 is the safe max
_MIN_CONTAINERS  = int(os.getenv("MODAL_MIN_CONTAINERS",  "0"))
_MAX_NEW_TOKENS  = int(os.getenv("MODAL_MAX_NEW_TOKENS",  "512"))
_SCALEDOWN_SECS  = int(os.getenv("MODAL_SCALEDOWN_SECS",  "300"))

# ── Volumes & paths ────────────────────────────────────────────────────────────
adapters_volume = modal.Volume.from_name("bids-eye-weights",  create_if_missing=True)
metadata_volume = modal.Volume.from_name("bids-eye-metadata", create_if_missing=True)

ADAPTERS_PATH    = "/adapters"
METADATA_PATH    = "/metadata"
NAME_INDEX_FILE  = f"{METADATA_PATH}/name_index.json"

HF_MODEL = "defog/sqlcoder-7b-2"

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install torch first, isolated, so its version is not pulled down by other
    # packages.  The extra-index-url ensures pip finds CUDA-enabled wheels.
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0,<2.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
        force_build=True,
    )
    .pip_install(
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "einops",
        "huggingface_hub>=0.22.0",
        # Preprocessing (Gemini) + RAG
        "google-genai>=0.8.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "rapidfuzz>=3.0.0",
        # DB query
        "sqlalchemy>=2.0",
        "psycopg2-binary>=2.9",
    )
    # Pre-download model weights at image build time so cold starts don't hit HF.
    # Must come before add_local_file calls (Modal rule: run_commands before local files).
    .run_commands(
        f"python -c \""
        f"from transformers import AutoModelForCausalLM, AutoTokenizer; "
        f"AutoTokenizer.from_pretrained('{HF_MODEL}', trust_remote_code=True); "
        f"AutoModelForCausalLM.from_pretrained('{HF_MODEL}', trust_remote_code=True)"
        f"\""
    )
    # Local source files — added last so changes don't invalidate the model cache layer
    .add_local_file("synthetic_data_generation_and_train/constants.py", "/app/constants.py")
    .add_local_file("synthetic_data/sql_expander.py",                   "/app/sql_expander.py")
    .add_local_file("synthetic_data/value_mappings.py",                 "/app/value_mappings.py")
    .add_local_file("RAG/value_mappings.yaml",                    "/app/value_mappings.yaml")
    .add_local_file("RAG/join_paths.yaml",                        "/app/join_paths.yaml")
    .add_local_file("RAG/yaml_to_llamaindex.py",                  "/app/yaml_to_llamaindex.py")
    .add_local_file("RAG/join_registry.py",                       "/app/join_registry.py")
    .add_local_file("RAG/retriever.py",                           "/app/retriever.py")
    .add_local_file("LLM_preprocessor/preprocess.py",             "/app/preprocess.py")
    .add_local_file("modal_app/few_shot_examples.json",           "/app/few_shot_examples.json")
)

app = modal.App("bids-eye", image=image)

# ── QueryFamily → training-data family name mapping ───────────────────────────
# preprocess.py uses enum values like "subject_multimodal"; training data uses
# "subject_multimodal_query".  This table normalises the mismatch.
_FAMILY_ALIASES: Dict[str, str] = {
    "subject_multimodal": "subject_multimodal_query",
    "metadata_query":     "json_query",
    "json_numeric_query": "json_numeric_query",  # already matches
}

# Fallback families to pull from when the requested family has < 5 examples
_FALLBACK_FAMILIES = ["combined_filter", "concept_query", "scan_filter"]


# ── SQL helpers ────────────────────────────────────────────────────────────────

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)

_FALLBACK_SQL = (
    "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
    "       d.source_type, d.remote_url, d.validation_status,\n"
    "       COUNT(DISTINCT o.subject) AS subject_count\n"
    "FROM bids_datasets d\n"
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
    "GROUP BY d.id\n"
    "ORDER BY d.name"
)


def _extract_sql(raw: str) -> str:
    m = re.search(r"\[SQL\]\s*(.*?)(?:\[/SQL\]|$)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if re.match(r"(?i)select\b", candidate):
            return candidate
    m = _SQL_FENCE.search(raw)
    if m:
        candidate = m.group(1).strip()
        if re.match(r"(?i)select\b", candidate):
            return candidate
    stripped = raw.strip()
    if re.match(r"(?i)select\b", stripped):
        return stripped.split("\n\n")[0].strip()
    return _FALLBACK_SQL


def _apply_pagination(sql: str, limit: Optional[int], offset: Optional[int]) -> str:
    # Always strip any LIMIT/OFFSET the LLM generated (avoids "50 subjects" → LIMIT 50).
    sql = re.sub(r"\s+LIMIT\s+\d+",  "", sql, flags=re.IGNORECASE).rstrip()
    sql = re.sub(r"\s+OFFSET\s+\d+", "", sql, flags=re.IGNORECASE).rstrip()
    # Re-add caller-specified pagination only when explicitly requested.
    if limit is not None:
        sql += f"\nLIMIT {limit}"
        if offset:
            sql += f"\nOFFSET {offset}"
    return sql


# ── SQL normalisation ──────────────────────────────────────────────────────────
# These are the columns the backend reads from every result row.
# Critical (accessed via r["col"], raise KeyError if absent): id, name, dataset_type, source_type
# Optional (accessed via r.get("col")): the rest
_REQUIRED_SELECT: List[tuple] = [
    ("d.id",                                       r"\bd\.id\b"),
    ("d.name",                                     r"\bd\.name\b"),
    ("d.accession_id",                             r"\bd\.accession_id\b"),
    ("d.bids_version",                             r"\bd\.bids_version\b"),
    ("d.dataset_type",                             r"\bd\.dataset_type\b"),
    ("d.source_type",                              r"\bd\.source_type\b"),
    ("d.remote_url",                               r"\bd\.remote_url\b"),
    ("d.validation_status",                        r"\bd\.validation_status\b"),
    ("COUNT(DISTINCT o.subject) AS subject_count", r"COUNT\s*\(\s*DISTINCT\s+o\d*\.subject\s*\)"),
]

_MANDATORY_COLS_SQL = (
    "d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
    "       d.source_type, d.remote_url, d.validation_status,\n"
    "       COUNT(DISTINCT o.subject) AS subject_count"
)


def _outer_select_span(sql: str):
    """Return (cols_start, from_start) character indices for the outermost SELECT clause.

    cols_start is just after 'SELECT [DISTINCT]'.
    from_start is where the top-level FROM keyword begins.
    Returns (None, None) if the structure can't be found.
    """
    m = re.match(r"\s*SELECT\s+(?:DISTINCT\s+)?", sql, re.IGNORECASE)
    if not m:
        return None, None
    cols_start = m.end()
    depth = 0
    i = cols_start
    while i < len(sql) - 3:
        c = sql[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            word = sql[i:i + 4].upper()
            if word == "FROM" and (i == 0 or not sql[i - 1].isalnum() and sql[i - 1] != "_"):
                return cols_start, i
        i += 1
    return cols_start, None


_BAD_OUTER_FILTER = re.compile(
    r'(?:AND\s+)?o\d*\.(task|datatype|suffix)\s*=\s*\'[^\']*\'',
    re.IGNORECASE,
)


def _strip_outer_scan_filters(sql: str) -> str:
    """Remove o.task/datatype/suffix equality filters from the outer SELECT query.

    These columns store canonical codes and must only be filtered inside EXISTS
    subqueries.  A direct `WHERE o.task = 'made_up_code'` always returns 0 rows.
    """
    out: list[str] = []
    depth = 0
    pos = 0
    while pos < len(sql):
        ch = sql[pos]
        if ch == '(':
            depth += 1
            out.append(ch)
            pos += 1
        elif ch == ')':
            depth -= 1
            out.append(ch)
            pos += 1
        elif depth == 0:
            m = _BAD_OUTER_FILTER.match(sql, pos)
            if m:
                print(f"[bids-eye] Stripped bad outer scan filter: {m.group(0).strip()}")
                pos = m.end()
            else:
                out.append(ch)
                pos += 1
        else:
            out.append(ch)
            pos += 1
    result = ''.join(out)
    result = re.sub(r'\bWHERE\s+(GROUP|ORDER|HAVING|LIMIT)\b', r'\1', result, flags=re.IGNORECASE)
    return result


def normalize_sql(sql: str) -> str:
    """Apply deterministic post-generation fixes before execution.

    0. Strip any direct outer-query filters on canonical-code columns.
    1. Fix COUNT without DISTINCT on subject columns.
    2. Expand bare SELECT * → mandatory column list.
    3. Inject any missing mandatory SELECT columns (skips CTEs — too complex to patch safely).
    4. Inject GROUP BY d.id when the clause is completely absent.
    """
    # 0. Strip hallucinated outer o.task/datatype/suffix filters
    sql = _strip_outer_scan_filters(sql)

    # 1. COUNT(o.subject) → COUNT(DISTINCT o.subject)  (handles o, o2, o3, …)
    sql = re.sub(
        r"\bCOUNT\s*\(\s*(?!DISTINCT\s)(o\d*\.subject)\s*\)",
        r"COUNT(DISTINCT \1)",
        sql,
        flags=re.IGNORECASE,
    )

    # 2. SELECT * → full mandatory column list
    if re.match(r"\s*SELECT\s+(?:DISTINCT\s+)?\*\s+FROM\b", sql, re.IGNORECASE):
        sql = re.sub(
            r"(SELECT\s+(?:DISTINCT\s+)?)\*",
            r"\g<1>" + _MANDATORY_COLS_SQL,
            sql,
            count=1,
            flags=re.IGNORECASE,
        )
        return sql  # GROUP BY check below still runs on the rewritten SQL in step 4

    # 3. Inject missing mandatory columns (skip CTEs — parsing them safely is fragile)
    if not re.match(r"\s*WITH\b", sql, re.IGNORECASE):
        cols_start, from_start = _outer_select_span(sql)
        if cols_start is not None and from_start is not None:
            cols_clause = sql[cols_start:from_start]
            missing = [
                col for col, pattern in _REQUIRED_SELECT
                if not re.search(pattern, cols_clause, re.IGNORECASE)
            ]
            if missing:
                prefix = ", ".join(missing) + ",\n       "
                sql = sql[:cols_start] + prefix + sql[cols_start:]

    # 4. Inject GROUP BY d.id if entirely absent
    if not re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE):
        tail = re.search(r"\b(?:ORDER\s+BY|LIMIT|OFFSET)\b", sql, re.IGNORECASE)
        if tail:
            insert_at = tail.start()
            sql = sql[:insert_at].rstrip() + "\nGROUP BY d.id\n" + sql[insert_at:]
        else:
            sql = sql.rstrip("; \n") + "\nGROUP BY d.id"

    return sql


# ── Few-shot selection ─────────────────────────────────────────────────────────

def _pick_examples(
    pool: Dict[str, List[dict]],
    query_family: str,
    n: int = 5,
    seed: str = "",
) -> List[dict]:
    """
    Return *n* examples for *query_family* from *pool*.

    - Uses the query text as a hash seed for deterministic but varied selection
      (same question always gets same examples; different questions vary).
    - Falls back to sibling families when the pool for the requested family is small.
    """
    family_key = _FAMILY_ALIASES.get(query_family, query_family)

    candidates = list(pool.get(family_key, []))

    # Pad from fallback families if needed
    if len(candidates) < n:
        for fb in _FALLBACK_FAMILIES:
            if fb == family_key:
                continue
            candidates.extend(pool.get(fb, []))
            if len(candidates) >= n:
                break

    if not candidates:
        return []

    # Deterministic shuffle via question hash
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    rng_candidates = candidates[:]
    # Fisher-Yates with seeded LCG
    for i in range(len(rng_candidates) - 1, 0, -1):
        j = h % (i + 1)
        rng_candidates[i], rng_candidates[j] = rng_candidates[j], rng_candidates[i]
        h = (h * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

    return rng_candidates[:n]


def _format_examples(examples: List[dict]) -> str:
    if not examples:
        return ""
    lines = ["### Examples",
             "Here are similar queries and their correct SQL:\n"]
    for ex in examples:
        lines.append(f"[QUESTION]{ex['question']}[/QUESTION]")
        lines.append(f"[SQL]\n{ex['sql'].strip()}\n[/SQL]\n")
    return "\n".join(lines)


# ── DB execution ───────────────────────────────────────────────────────────────

def _execute_query(sql: str, db_url: str) -> List[Dict[str, Any]]:
    """Run *sql* against *db_url* and return rows as a list of dicts."""
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]


def _correct_sql_with_gemini(
    sql: str,
    db_error: str,
    question: str,
    schema_ddl: str,
    api_key: str,
) -> str:
    """
    Ask Gemini to fix a SQL query that failed with a PostgreSQL error.

    Returns the corrected SQL string (plain SELECT, no fences).
    Falls back to the original sql if Gemini returns something unparseable.
    """
    from google import genai
    from google.genai import types

    prompt = (
        "A SQL query for PostgreSQL failed with an error. "
        "Fix it and return ONLY the corrected SQL — no explanation, no markdown fences.\n\n"
        f"Original question:\n{question}\n\n"
        f"Database schema:\n{schema_ddl}\n\n"
        f"Failed SQL:\n{sql}\n\n"
        f"PostgreSQL error:\n{db_error}"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    corrected = _extract_sql(response.text.strip())
    # If extraction fell through to the fallback, return the original sql instead
    return corrected if corrected != _FALLBACK_SQL else sql


# ── Modal class ────────────────────────────────────────────────────────────────

@app.cls(
    gpu="A10G",
    volumes={
        ADAPTERS_PATH: adapters_volume,
        METADATA_PATH: metadata_volume,
    },
    secrets=[modal.Secret.from_name("bids-eye-gemini")],
    timeout=600,
    scaledown_window=_SCALEDOWN_SECS,
    min_containers=_MIN_CONTAINERS,
)
class TextToSQLModel:
    """
    Full Text-to-SQL pipeline: preprocessing → RAG → few-shot → SQLCoder → DB.

    Primary entry point:  TextToSQLModel.run(question)
    Legacy entry point:   TextToSQLModel.generate(question)  [SQLCoder only, no preprocessing]
    """

    # ── Container startup ─────────────────────────────────────────────────────
    @modal.enter()
    def load_model(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ── /app on sys.path so all copied modules are importable ──────────────
        sys.path.insert(0, "/app")
        from constants import SCHEMA_DDL
        from sql_expander import expand_sql
        from preprocess import run_pipeline
        from retriever import MetadataRetriever

        self._schema_ddl = SCHEMA_DDL
        self._expand_sql = expand_sql
        self._run_pipeline = run_pipeline
        self._gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        # ── Few-shot pool ──────────────────────────────────────────────────────
        few_shot_path = Path("/app/few_shot_examples.json")
        if few_shot_path.exists():
            with open(few_shot_path) as fh:
                self._few_shot_pool: Dict[str, List[dict]] = json.load(fh)
            total = sum(len(v) for v in self._few_shot_pool.values())
            print(f"[bids-eye] Loaded {total} few-shot examples "
                  f"across {len(self._few_shot_pool)} families")
        else:
            self._few_shot_pool = {}
            print("[bids-eye] WARNING: few_shot_examples.json not found — "
                  "few-shot injection disabled")

        # ── RAG retriever ──────────────────────────────────────────────────────
        if Path(NAME_INDEX_FILE).exists():
            try:
                self._retriever: Optional[Any] = MetadataRetriever(NAME_INDEX_FILE)
                print(f"[bids-eye] MetadataRetriever loaded from {NAME_INDEX_FILE}")
            except Exception as exc:
                self._retriever = None
                print(f"[bids-eye] WARNING: could not load name index: {exc}")
        else:
            self._retriever = None
            print(f"[bids-eye] No name index at {NAME_INDEX_FILE} — "
                  "author/funding RAG disabled. "
                  "Build with: python RAG/build_name_index.py")

        # ── SQLCoder ───────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        adapter_config = Path(ADAPTERS_PATH) / "adapter_config.json"
        if adapter_config.exists():
            self.model = PeftModel.from_pretrained(base, ADAPTERS_PATH)
            print(f"[bids-eye] Loaded LoRA adapters from {ADAPTERS_PATH}")
        else:
            self.model = base
            print(f"[bids-eye] WARNING: no adapter_config.json at {ADAPTERS_PATH} "
                  "— running base model without LoRA fine-tuning. "
                  "Upload adapters with: modal volume put bids-eye-weights <local_dir> /")
        self.model.eval()

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_prompt(self, question: str, examples: List[dict]) -> str:
        """
        Build the SQLCoder prompt.

        Structure:
          ### Task
          ### Instructions
          ### Database Schema
          ### Examples   ← injected few-shot block (if available)
          ### Answer     ← model fills from here
        """
        instructions = (
            "- CRITICAL — VOCABULARY MISS terms: If [VOCABULARY MISS] appears in the question, "
            "those terms have NO canonical code. Use ONLY the ILIKE clause shown in that block "
            "(copy it verbatim into WHERE). NEVER set o.task, o.datatype, o.suffix, or "
            "p.diagnosis equal to a VOCABULARY MISS term — those columns hold canonical codes only.\n"
            "- Only use tables and columns present in the schema below.\n"
            "- Always SELECT: d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, "
            "d.source_type, d.remote_url, d.validation_status, COUNT(DISTINCT o.subject) AS subject_count\n"
            "- Always: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "- Always: GROUP BY d.id\n"
            "- If [Resolved DB filters] are provided in the question, copy each EXISTS subquery "
            "VERBATIM into the WHERE clause — do not change column names, operators, or values\n"
            "- Use ILIKE '%term%' for case-insensitive text search\n"
            "- Do NOT add LIMIT unless the question explicitly asks for 'top N' results\n"
            "- For any file or participant filter NOT already covered by [Resolved DB filters], "
            "use an EXISTS (...) subquery in the WHERE clause\n"
            "- For threshold filters on aggregated counts (e.g. 'more than 5 subjects', "
            "'at least 100 participants'), use a HAVING clause — not WHERE:\n"
            "  WRONG: WHERE COUNT(DISTINCT o.subject) > 5\n"
            "  RIGHT: HAVING COUNT(DISTINCT o.subject) > 5\n"
            "- For participant-count thresholds broken down by a criterion (e.g. 'more male "
            "than female participants'), use correlated subqueries in HAVING:\n"
            "  HAVING (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id "
            "AND p.sex = 'M') > "
            "(SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id "
            "AND p.sex = 'F')\n"
            "- NEVER JOIN bids_participants directly alongside bids_objects — "
            "this creates a cross-product. Always use EXISTS for participant filters:\n"
            "  WRONG: LEFT JOIN bids_participants p ON p.dataset_id = d.id\n"
            "  RIGHT: EXISTS (SELECT 1 FROM bids_participants p2 "
            "WHERE p2.dataset_id = d.id AND p2.diagnosis = '...')\n"
            "- NEVER use COUNT(o.subject): always COUNT(DISTINCT o.subject) AS subject_count\n"
            "- In EXISTS subqueries, always correlate the subquery table back to the outer "
            "query using dataset_id = d.id "
            "(e.g. WHERE p2.dataset_id = d.id or WHERE o2.dataset_id = d.id)\n"
        )

        examples_block = _format_examples(examples)
        sep = "\n\n" if examples_block else ""

        return (
            f"### Task\n"
            f"Generate a SQL query to answer [QUESTION]{question}[/QUESTION]\n\n"
            f"### Instructions\n"
            f"{instructions}\n\n"
            f"### Database Schema\n"
            f"The query will run on a database with the following schema:\n"
            f"```sql\n{self._schema_ddl}\n```\n\n"
            f"{examples_block}{sep}"
            f"### Answer\n"
            f"Given the database schema, here is the SQL query that "
            f"answers [QUESTION]{question}[/QUESTION]\n"
            f"[SQL]\n"
        )

    # ── SQL execution with Gemini correction loop ─────────────────────────────

    def _execute_with_correction(
        self,
        sql: str,
        db_url: str,
        question: str,
        max_corrections: int = 2,
    ) -> tuple:
        """
        Execute *sql* against *db_url*, retrying with Gemini correction on failure.

        Parameters
        ----------
        sql             : initial SQL to try
        db_url          : PostgreSQL connection URL
        question        : augmented question (for context in correction prompt)
        max_corrections : how many Gemini correction rounds to attempt (default 2)

        Returns
        -------
        (final_sql, rows, corrections, error)
          final_sql   : the SQL that ultimately succeeded (or the last attempt)
          rows        : list-of-dicts if execution succeeded, else None
          corrections : list of {"attempt": N, "sql": ..., "error": ...}
                        one entry per failed attempt
          error       : None on success, error string after all attempts exhausted
        """
        corrections: List[Dict[str, Any]] = []
        current_sql = sql

        for attempt in range(max_corrections + 1):
            try:
                rows = _execute_query(current_sql, db_url)
                return current_sql, rows, corrections, None
            except Exception as exc:
                db_err = str(exc)
                corrections.append({
                    "attempt": attempt + 1,
                    "sql":     current_sql,
                    "error":   db_err,
                })
                print(f"[bids-eye] SQL attempt {attempt + 1} failed: {db_err[:300]}")

                if attempt < max_corrections and self._gemini_key:
                    print(f"[bids-eye] Sending to Gemini for correction "
                          f"(round {attempt + 1}/{max_corrections}) ...")
                    try:
                        current_sql = _correct_sql_with_gemini(
                            current_sql, db_err, question,
                            self._schema_ddl, self._gemini_key,
                        )
                        print(f"[bids-eye] Corrected SQL:\n{current_sql}")
                    except Exception as gemini_exc:
                        print(f"[bids-eye] Gemini correction call failed: {gemini_exc}")
                        break

        final_error = (
            f"All {len(corrections)} SQL attempt(s) failed. "
            f"Last error: {corrections[-1]['error']}"
        )
        return current_sql, None, corrections, final_error

    # ── Low-level inference ────────────────────────────────────────────────────

    def _infer(self, prompt: str, max_new_tokens: int = 512) -> str:
        import torch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to("cuda")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ── Primary entry point: full pipeline ────────────────────────────────────

    @modal.method()
    def run(
        self,
        question: str,
        db_url: Optional[str] = None,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        n_examples: int = 5,
    ) -> dict:
        """
        Full pipeline: preprocess → RAG → few-shot → SQLCoder → optionally query DB.

        Parameters
        ----------
        question       : natural-language question from the user
        db_url         : optional PostgreSQL URL; if given, executes the SQL and
                         returns rows in the result
        max_new_tokens : token budget for the SQLCoder generation step
        limit / offset : pagination; appended to generated SQL when provided
        n_examples     : number of few-shot examples to inject (default 5)

        Returns
        -------
        {
          "sql":              final SQL string (possibly Gemini-corrected)
          "sql_corrections":  list of {"attempt", "sql", "error"} for each failed attempt
          "query_plan":       structured QueryPlan dict (from preprocessing)
          "augmented_question": question with resolved DB codes injected
          "rag_resolved":     {field: {term: [canonical_codes]}} from RAG
          "rows":             list of result dicts if db execution succeeded, else null
          "error":            error message string if something failed, else null
        }
        """
        error: Optional[str] = None
        query_plan_dict: Optional[dict] = None
        augmented_question = question
        rag_resolved: Dict = {}
        sql = _FALLBACK_SQL
        rows: Optional[List] = None
        sql_corrections: List[Dict[str, Any]] = []
        query_family = "combined_filter"

        # ── 1. Preprocess + RAG ────────────────────────────────────────────────
        if not self._gemini_key:
            error = ("GEMINI_API_KEY not set — preprocessing skipped. "
                     "Set via Modal secret 'bids-eye-gemini'.")
            print(f"[bids-eye] WARNING: {error}")
        else:
            try:
                augmented_plan = self._run_pipeline(
                    question,
                    retriever=self._retriever,
                    api_key=self._gemini_key,
                )
                augmented_question = augmented_plan.augmented_question
                query_plan_dict = augmented_plan.plan.model_dump()
                query_family = augmented_plan.plan.query_family.value
                rag_resolved = {
                    r.field.value: r.term_to_codes
                    for r in augmented_plan.resolved
                }
            except Exception as exc:
                error = f"Preprocessing failed: {exc}"
                print(f"[bids-eye] WARNING: {error}")

        # ── 2. Few-shot selection ──────────────────────────────────────────────
        examples = _pick_examples(
            self._few_shot_pool,
            query_family,
            n=n_examples,
            seed=question,
        )

        # ── 3. SQLCoder inference ──────────────────────────────────────────────
        prompt = self._build_prompt(augmented_question, examples)
        raw = self._infer(prompt, max_new_tokens=max_new_tokens)
        sql = _apply_pagination(
            normalize_sql(self._expand_sql(_extract_sql(raw))),
            limit,
            offset,
        )

        # ── 4. DB execution with Gemini correction loop ────────────────────────
        if db_url:
            sql, rows, sql_corrections, db_error = self._execute_with_correction(
                sql, db_url, augmented_question,
            )
            if db_error:
                error = (error + "; " + db_error) if error else db_error

        import json as _json
        result = {
            "sql":                sql,
            "sql_corrections":    sql_corrections,
            "query_plan":         query_plan_dict,
            "augmented_question": augmented_question,
            "rag_resolved":       rag_resolved,
            "rows":               rows,
            "error":              error,
        }
        # Serialize through JSON to strip any custom types (enums, Pydantic models)
        # from preprocess/retriever that would fail to deserialize in the local env.
        return _json.loads(_json.dumps(result, default=str))

    # ── Legacy entry point: SQLCoder only (no preprocessing) ──────────────────

    @modal.method()
    def generate(
        self,
        question: str,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> dict:
        """
        SQLCoder inference only — no Gemini preprocessing, no RAG.
        Kept for backward compatibility and for cases where the caller
        has already done its own preprocessing.

        Returns: {"sql": ..., "augmented_question": ..., "context_hints": {}}
        """
        prompt = self._build_prompt(question, examples=[])
        raw = self._infer(prompt, max_new_tokens=max_new_tokens)
        sql = _apply_pagination(normalize_sql(self._expand_sql(_extract_sql(raw))), limit, offset)
        return {
            "sql":                sql,
            "augmented_question": question,
            "context_hints":      {},
        }


# ── Local entrypoint (modal run modal_app/app.py --question "...") ─────────────

@app.local_entrypoint()
def main(
    question: str = "Find all fMRI datasets with more than 50 subjects",
    db_url: str = "",
):
    """
    modal run modal_app/app.py --question "..." [--db-url "postgresql://..."]
    """
    import json

    model = TextToSQLModel()
    result = model.run.remote(question, db_url=db_url or None)

    print("\n── Augmented question ──────────────────────────────────────────")
    print(result["augmented_question"])

    if result.get("query_plan"):
        print("\n── Query plan ──────────────────────────────────────────────────")
        print(json.dumps(result["query_plan"], indent=2))

    print("\n── Generated SQL ───────────────────────────────────────────────")
    print(result["sql"])

    if result.get("sql_corrections"):
        print("\n── SQL correction history ──────────────────────────────────────")
        for c in result["sql_corrections"]:
            print(f"\n  Attempt {c['attempt']} failed:")
            print(f"  Error : {c['error'][:300]}")
            print(f"  SQL   :\n{c['sql']}")

    if result.get("rows") is not None:
        rows = result["rows"]
        print(f"\n── Results ({len(rows)} row{'s' if len(rows) != 1 else ''}) ──────────────────────────────────────")
        if rows:
            # Print as a simple table
            cols = list(rows[0].keys())
            col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
            header = "  ".join(c.ljust(col_widths[c]) for c in cols)
            print(header)
            print("-" * len(header))
            for row in rows[:50]:  # cap at 50 rows in terminal output
                print("  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in cols))
            if len(rows) > 50:
                print(f"  ... and {len(rows) - 50} more rows")
        else:
            print("  (no rows returned)")

    if result.get("error"):
        print(f"\n── Error ───────────────────────────────────────────────────────")
        print(result["error"])
