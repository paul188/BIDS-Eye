"""
services/translation_store.py
-----------------------------
Translations of questions that were asked before, so the same question does not
pay for the two model calls again.

Only exact matches: the key is the normalised text of the question
(``normalize_question``). Nothing is matched by similarity.

An entry holds what the two model stages produced: the augmented plan
(stage 1 + 2) and, when the plan was not compiled, the SQL the model wrote. It
does not hold the final SQL. On a hit ``services.text_to_sql._run_local`` runs
the plan compiler, the SQL rewriter and the uncertain tier as for a new
question, so a change to those applies to stored entries at once.

Every entry has a review status:

  unreviewed    stored from a search, nobody looked at it. Used while what it
                was made from is unchanged (below) and for at most
                TRANSLATION_STORE_TTL_DAYS (default 7); after that the next
                search translates anew and replaces it.
  verified      a person approved it (review_translations.py). Does not
                expire, and is not replaced by a new translation.
  needs_review  was verified, then the schema or the vocabulary and criteria
                changed. Not used until someone approves it again.
  rejected      a person found it wrong. Never used and never replaced, so the
                question is translated anew on every search.

What an entry was made from:

  schema       the optional columns and search capabilities of the database
  vocabulary   the data files in RAG/ and RAG/criteria.py
  pipeline     all code, prompts, examples and vocabulary files that turn a
               question into a plan and raw SQL, and the model list
  data stamp   the number of datasets (a crawl changes the vocabulary the
               resolution step reads from the database)

Every entry also records the git commit of the code that made it
(``code_commit()``), so that after a fix one can list what an older version
stored: `review_translations.py commits` and `list --commit`.

An unreviewed entry depends on all four. A verified one only on schema and
vocabulary: its SQL was approved as the meaning of the question, which a
changed prompt or a new dataset does not undo.

Entries live in a dict in this process, read from the table
``query_translations`` at startup and again every five minutes (so a review
reaches the running backend), and written there after each stored search. Like
services/query_cache.py this assumes one backend worker: a second worker would
only see entries written before it started.

Switch: TRANSLATION_STORE=true (default off).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = Path(__file__).resolve().parents[1]

# Everything that decides the plan and the raw SQL of a question. The SQL
# rewriter and the uncertain tier are not here: they run again on every hit.
_PIPELINE_DIRS = (_ROOT / "LLM_preprocessor", _ROOT / "RAG")
_PIPELINE_SUFFIXES = (".py", ".yaml", ".yml", ".json")
_PIPELINE_FILES = (
    _BACKEND / "services" / "text_to_sql.py",
    _BACKEND / "services" / "participant_query_compiler.py",
    _BACKEND / "constants.py",
    _BACKEND / "few_shot_examples.json",
)
_PIPELINE_ENV = ("GEMINI_MODEL_CASCADE", "ANTHROPIC_FALLBACK_MODEL")

_STAMP_REFRESH_SECONDS = 300.0

# Commit of translations recorded before commits were tracked.
PRE_TRACKING = "pre-tracking"

UNREVIEWED = "unreviewed"
VERIFIED = "verified"
NEEDS_REVIEW = "needs_review"
REJECTED = "rejected"
STATUSES = (UNREVIEWED, VERIFIED, NEEDS_REVIEW, REJECTED)

# Vocabulary and criteria: what a verified entry depends on besides the schema.
_VOCABULARY_SUFFIXES = (".yaml", ".yml", ".json")
_VOCABULARY_FILES = (_ROOT / "RAG" / "criteria.py",)


def enabled() -> bool:
    return os.getenv("TRANSLATION_STORE", "false").strip().lower() in ("1", "true", "yes", "on")


def _ttl() -> timedelta:
    return timedelta(days=float(os.getenv("TRANSLATION_STORE_TTL_DAYS", "7")))


def _max_entries() -> int:
    return int(os.getenv("TRANSLATION_STORE_MAX_ENTRIES", "5000"))


# ── the key ─────────────────────────────────────────────────────────────────────

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize_question(question: str) -> str:
    """The text two questions must share to count as the same question.

    Case, spacing, the kind of quote marks and punctuation at the end are
    ignored. Every word and number stays: "with" and "without", "at least 40"
    and "at most 40" must never share a key.
    """
    text = unicodedata.normalize("NFKC", question or "").translate(_QUOTES).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".?! ").strip()


def question_key(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


# ── what an entry depends on ────────────────────────────────────────────────────

def _digest_of(paths: List[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _pipeline_files_digest() -> str:
    paths: List[Path] = [p for p in _PIPELINE_FILES if p.is_file()]
    for directory in _PIPELINE_DIRS:
        if directory.is_dir():
            paths.extend(p for p in directory.iterdir()
                         if p.is_file() and p.suffix in _PIPELINE_SUFFIXES)
    return _digest_of(paths)


@lru_cache(maxsize=1)
def vocabulary_fingerprint() -> str:
    """Hash of the vocabulary files and the criteria (read once per process)."""
    paths: List[Path] = [p for p in _VOCABULARY_FILES if p.is_file()]
    rag = _ROOT / "RAG"
    if rag.is_dir():
        paths.extend(p for p in rag.iterdir()
                     if p.is_file() and p.suffix in _VOCABULARY_SUFFIXES)
    return _digest_of(paths)


def pipeline_fingerprint() -> str:
    """Hash of the files and settings that turn a question into plan and raw SQL.

    The files are read once per process; a deploy or a reload starts a new one.
    """
    digest = hashlib.sha256(_pipeline_files_digest().encode("ascii"))
    for name in _PIPELINE_ENV:
        digest.update(f"\0{name}={os.getenv(name, '')}".encode("utf-8"))
    return digest.hexdigest()


def schema_fingerprint() -> str:
    """Hash of the optional columns and search capabilities of this database."""
    from services import schema_capabilities as caps
    parts = sorted(caps.available_columns()) + [
        f"fts={caps.has_fts()}", f"paper_fts={caps.has_paper_fts()}",
        f"exact_fts={caps.has_exact_fts()}", f"exact_paper_fts={caps.has_exact_paper_fts()}",
        f"word_forms={caps.has_word_forms()}",
    ]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def code_commit() -> str:
    """The git commit of the running code.

    In order: BIDSEYE_COMMIT; the file DEPLOYED_COMMIT next to the code (both
    are put into the image when it is built, see backend/Dockerfile); git, for
    a checkout, with "-dirty" when files differ from the commit; else
    "unknown".
    """
    named = (os.getenv("BIDSEYE_COMMIT") or "").strip()
    if named:
        return named[:64]
    try:
        deployed = (_ROOT / "DEPLOYED_COMMIT").read_text().split()
        if deployed:
            return deployed[0][:64]
    except OSError:
        pass
    try:
        import subprocess

        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(_ROOT), *args], capture_output=True,
                                  text=True, timeout=5, check=True).stdout.strip()
        commit = git("rev-parse", "--short=12", "HEAD")
        if commit:
            return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except Exception:  # noqa: BLE001 — no git, or not a checkout
        pass
    return "unknown"


_data_stamp: Optional[str] = None
_data_stamp_read_at: float = 0.0


def data_stamp() -> Optional[str]:
    """The catalogue state entries are tied to; None until it was read."""
    return _data_stamp


def set_data_stamp(stamp: Optional[str]) -> None:
    global _data_stamp, _data_stamp_read_at
    _data_stamp = stamp
    _data_stamp_read_at = time.monotonic()


# ── entries ─────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StoredTranslation:
    question_norm: str
    plan: Dict[str, Any]              # AugmentedQueryPlan, as JSON
    model_sql: Optional[str]          # None when the plan was compiled
    pipeline_fp: str
    vocabulary_fp: str
    schema_fp: str
    data_stamp: str
    status: str = UNREVIEWED
    created_at: datetime = field(default_factory=_now)
    # For the reviewer: the SQL the search ran and how many datasets it found.
    final_sql: Optional[str] = None
    result_total: Optional[int] = None
    code_commit: str = field(default_factory=code_commit)

    @property
    def key(self) -> str:
        return hashlib.sha256(self.question_norm.encode("utf-8")).hexdigest()

    def stale_reason(self) -> Optional[str]:
        """Why this entry may not be used now; None when it may."""
        if self.status == REJECTED:
            return "rejected in review"
        if self.status == NEEDS_REVIEW:
            return "awaiting review"
        changed = self.changed_since_review()
        if changed is not None or self.status == VERIFIED:
            return changed
        if self.pipeline_fp != pipeline_fingerprint():
            return "pipeline changed"
        if self.data_stamp != data_stamp():
            return "catalogue changed"
        if _now() - self.created_at > _ttl():
            return "expired"
        return None

    def changed_since_review(self) -> Optional[str]:
        """The change that makes a reviewed entry need a new review, if any."""
        if self.schema_fp != schema_fingerprint():
            return "schema changed"
        if self.vocabulary_fp != vocabulary_fingerprint():
            return "vocabulary or criteria changed"
        return None


class _Store:
    def __init__(self) -> None:
        self._entries: Dict[str, StoredTranslation] = {}
        self._lock = threading.Lock()

    def get(self, question: str) -> Optional[StoredTranslation]:
        key = question_key(question)
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return None
        reason = entry.stale_reason()
        if reason is not None:
            log.info("translation store: entry not used (%s)", reason)
            return None
        return entry

    def reviewed(self, key: str) -> bool:
        """True when a person decided on this question's entry."""
        with self._lock:
            entry = self._entries.get(key)
        return entry is not None and entry.status != UNREVIEWED

    def replace_all(self, entries: List[StoredTranslation]) -> None:
        with self._lock:
            self._entries = {entry.key: entry for entry in entries}

    def put(self, entry: StoredTranslation) -> None:
        with self._lock:
            self._entries[entry.key] = entry
            overflow = len(self._entries) - _max_entries()
            if overflow > 0:
                oldest = sorted(self._entries, key=lambda k: self._entries[k].created_at)
                for key in oldest[:overflow]:
                    self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


store = _Store()


def lookup(question: str) -> Optional[StoredTranslation]:
    """The usable entry for this question, or None. Never raises."""
    if not enabled():
        return None
    try:
        return store.get(question)
    except Exception as exc:  # noqa: BLE001 — the store must never break a search
        log.warning("translation store: lookup failed (%s)", exc)
        return None


def candidate(question: str, augmented_plan: Any,
              model_sql: Optional[str]) -> Optional[StoredTranslation]:
    """An entry for a translation just made, to be stored once the search worked.

    None when the store is off or the catalogue state is unknown.
    """
    if not enabled() or data_stamp() is None or not normalize_question(question):
        return None
    try:
        # Topics are only needed while the augmented question is built.
        plan = augmented_plan.model_dump(mode="json", exclude={"topics"})
        return StoredTranslation(
            question_norm=normalize_question(question), plan=plan, model_sql=model_sql,
            pipeline_fp=pipeline_fingerprint(), vocabulary_fp=vocabulary_fingerprint(),
            schema_fp=schema_fingerprint(), data_stamp=data_stamp(),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("translation store: could not build an entry (%s)", exc)
        return None


# ── capture file ────────────────────────────────────────────────────────────────
# Any process that translates questions (a benchmark run, a script, the
# backend) appends each new translation to TRANSLATION_CAPTURE_FILE when that
# is set, whether the store is on or not. The file is one JSON object per line
# and is read by `review_translations.py import`, which checks every
# translation against the database before it stores it.

_capture_lock = threading.Lock()


def capture_file() -> Optional[str]:
    return os.getenv("TRANSLATION_CAPTURE_FILE") or None


def capture(question: str, augmented_plan: Any, model_sql: Optional[str]) -> None:
    """Append a new translation to the capture file, if one is set. Never raises."""
    path = capture_file()
    if not path:
        return
    try:
        import json
        line = json.dumps({
            "question": question,
            "plan": augmented_plan.model_dump(mode="json", exclude={"topics"}),
            "model_sql": model_sql,
            "code_commit": code_commit(),
            "captured_at": _now().isoformat(timespec="seconds"),
        })
        with _capture_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("translation store: translation not captured (%s)", exc)


_PLAN_KEYS = ("plan", "resolved", "scored_filters", "augmented_question", "word_forms")


def read_translations(path: Any) -> List[Dict[str, Any]]:
    """Translations from a capture file or from a recording of record_run.py.

    Returns {"question", "plan", "model_sql", "code_commit"} per translation;
    code_commit is None when the file does not say which code made it. For a
    question that appears more than once the last one counts.
    """
    import json
    text = Path(path).read_text(encoding="utf-8")
    found: Dict[str, Dict[str, Any]] = {}
    try:
        recording = json.loads(text)
    except ValueError:
        recording = None
    if isinstance(recording, dict) and "question" not in recording:
        # record_run.py: {question: augmented plan fields + model_sql [+ usage]}
        for question, saved in recording.items():
            found[normalize_question(question)] = {
                "question": question,
                "plan": {k: saved[k] for k in _PLAN_KEYS if k in saved},
                "model_sql": saved.get("model_sql"),
                "code_commit": saved.get("code_commit"),
            }
    else:
        for line in text.splitlines():
            if line.strip():
                item = json.loads(line)
                found[normalize_question(item["question"])] = {
                    "question": item["question"], "plan": item["plan"],
                    "model_sql": item.get("model_sql"),
                    "code_commit": item.get("code_commit"),
                }
    found.pop("", None)
    return list(found.values())


# ── database ────────────────────────────────────────────────────────────────────

def _from_row(row: Any) -> StoredTranslation:
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return StoredTranslation(
        question_norm=row.question_norm, plan=row.plan, model_sql=row.model_sql,
        pipeline_fp=row.pipeline_fp, vocabulary_fp=row.vocabulary_fp,
        schema_fp=row.schema_fp, data_stamp=row.data_stamp,
        status=row.status, created_at=created,
        final_sql=row.final_sql, result_total=row.result_total,
        code_commit=row.code_commit or PRE_TRACKING,
    )


def _table():
    from models import QueryTranslation
    return QueryTranslation.__table__


async def ensure_table(connection: Any) -> None:
    """Create the table, or add the columns a newer version of it has (Postgres)."""
    from sqlalchemy import text
    table = _table()
    await connection.run_sync(lambda sync: table.create(sync, checkfirst=True))
    await connection.execute(text(
        "ALTER TABLE query_translations ADD COLUMN IF NOT EXISTS code_commit VARCHAR(64)"))
    await connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_query_translations_code_commit "
        "ON query_translations (code_commit)"))


async def _read_data_stamp(connection: Any) -> str:
    from sqlalchemy import text
    count = (await connection.execute(text("SELECT count(*) FROM bids_datasets"))).scalar()
    return f"datasets={count}"


async def _read_entries(connection: Any) -> List[StoredTranslation]:
    from sqlalchemy import select
    return [_from_row(row) for row in (await connection.execute(select(_table()))).all()]


async def mark_for_review(connection: Any) -> int:
    """Set verified entries to needs_review when schema or vocabulary changed.

    Returns how many were marked. The caller commits.
    """
    from sqlalchemy import update
    table = _table()
    marked = 0
    for entry in await _read_entries(connection):
        reason = entry.changed_since_review() if entry.status == VERIFIED else None
        if reason is None:
            continue
        await connection.execute(
            update(table).where(table.c.question_key == entry.key)
            .values(status=NEEDS_REVIEW, review_reason=reason))
        marked += 1
    return marked


async def load(engine: Any) -> int:
    """Startup: read the catalogue state, mark what needs review, read all entries."""
    if not enabled():
        return 0
    async with engine.begin() as connection:
        await ensure_table(connection)
        set_data_stamp(await _read_data_stamp(connection))
        marked = await mark_for_review(connection)
        entries = await _read_entries(connection)
    store.replace_all(entries)
    log.info("translation store: %d entries loaded (%d newly marked for review), catalogue %s",
             len(store), marked, data_stamp())
    return len(store)


async def refresh(session: Any) -> None:
    """Every five minutes: re-read the catalogue state and the entries.

    The entries are re-read so that a review made with review_translations.py
    reaches the running backend.
    """
    if not enabled():
        return
    if _data_stamp is not None and time.monotonic() - _data_stamp_read_at < _STAMP_REFRESH_SECONDS:
        return
    try:
        stamp = await _read_data_stamp(session)
        entries = await _read_entries(session)
        set_data_stamp(stamp)
        store.replace_all(entries)
    except Exception as exc:  # noqa: BLE001
        log.warning("translation store: could not refresh from the database (%s)", exc)


async def remember(session: Any, entry: StoredTranslation) -> None:
    """Keep a new, unreviewed entry: in this process and, committed, in the table.

    An entry a person has decided on is never replaced by a search.
    """
    if store.reviewed(entry.key):
        return
    store.put(entry)
    from models import QueryTranslation
    try:
        async with session.begin_nested():
            await session.merge(QueryTranslation(
                question_key=entry.key, question_norm=entry.question_norm, status=entry.status,
                plan=entry.plan, model_sql=entry.model_sql, pipeline_fp=entry.pipeline_fp,
                vocabulary_fp=entry.vocabulary_fp, schema_fp=entry.schema_fp,
                data_stamp=entry.data_stamp, created_at=entry.created_at,
                final_sql=entry.final_sql, result_total=entry.result_total,
                code_commit=entry.code_commit, hits=0, review_reason=None,
            ))
        await session.commit()
    except Exception as exc:  # noqa: BLE001 — the entry still serves from memory
        log.warning("translation store: entry not written to the database (%s)", exc)


async def count_hit(session: Any, question: str) -> None:
    """Count a search answered from the store (the most used are reviewed first)."""
    from sqlalchemy import func, update
    table = _table()
    try:
        async with session.begin_nested():
            await session.execute(
                update(table).where(table.c.question_key == question_key(question))
                .values(hits=table.c.hits + 1, last_used_at=func.now()))
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("translation store: hit not counted (%s)", exc)


# ── review (review_translations.py) ─────────────────────────────────────────────

async def list_for_review(connection: Any, status: Optional[str] = None,
                          commit: Optional[str] = None) -> List[Any]:
    """Rows of the table, the most used first; *commit* is the start of a commit."""
    from sqlalchemy import select
    table = _table()
    query = select(table).order_by(table.c.hits.desc(), table.c.created_at)
    if status:
        query = query.where(table.c.status == status)
    if commit == PRE_TRACKING:
        query = query.where(table.c.code_commit.is_(None) | (table.c.code_commit == PRE_TRACKING))
    elif commit:
        query = query.where(table.c.code_commit.startswith(commit))
    return list((await connection.execute(query)).all())


async def count_by_commit(connection: Any) -> List[Any]:
    """(commit, status, entries, first stored, last stored), newest commit first."""
    from sqlalchemy import func, select
    table = _table()
    commit = func.coalesce(table.c.code_commit, PRE_TRACKING).label("code_commit")
    return list((await connection.execute(
        select(commit, table.c.status, func.count().label("entries"),
               func.min(table.c.created_at).label("first"), func.max(table.c.created_at).label("last"))
        .group_by(commit, table.c.status)
        .order_by(func.max(table.c.created_at).desc(), table.c.status))).all())


async def find(connection: Any, key_prefix: str) -> Any:
    """The one row whose key starts with *key_prefix*; LookupError otherwise."""
    from sqlalchemy import select
    table = _table()
    rows = (await connection.execute(
        select(table).where(table.c.question_key.startswith(key_prefix)))).all()
    if len(rows) != 1:
        raise LookupError(f"{len(rows)} entries start with {key_prefix!r}")
    return rows[0]


async def review(connection: Any, key_prefix: str, status: str, reviewer: str,
                 model_sql: Optional[str] = None) -> Any:
    """Record a person's decision on one entry. The caller commits.

    Approving ties the entry to the schema and vocabulary of now. *model_sql*
    replaces the SQL (in the form the SQL model writes it); a verified entry's
    SQL is used even when the plan could be compiled.
    """
    if status not in (VERIFIED, REJECTED):
        raise ValueError("a review ends in 'verified' or 'rejected'")
    from sqlalchemy import update
    table = _table()
    row = await find(connection, key_prefix)
    values: Dict[str, Any] = {"status": status, "reviewed_by": reviewer,
                              "reviewed_at": _now(), "review_reason": None}
    if status == VERIFIED:
        values.update(schema_fp=schema_fingerprint(), vocabulary_fp=vocabulary_fingerprint())
    if model_sql is not None:
        values["model_sql"] = model_sql
    await connection.execute(
        update(table).where(table.c.question_key == row.question_key).values(**values))
    return row


# ── import (review_translations.py import) ──────────────────────────────────────

async def import_translations(connection: Any, translations: List[Dict[str, Any]],
                              validate: Any) -> Dict[str, List[str]]:
    """Store captured translations as unreviewed entries. The caller commits.

    *validate* is awaited with (question, plan, model_sql) and returns
    (the SQL the search runs, datasets found); it raises when the translation
    cannot be used as it is. Stored are those that run and find something,
    the same rule as for a live search. They are tied to the pipeline, schema
    and catalogue of now, so import only what the current code would produce.
    An entry a person has decided on is left alone.

    The commit of an entry is that of the code that made the translation, as
    the file records it, never that of the importing code; a translation
    whose file names no commit gets PRE_TRACKING.

    Returns the questions per outcome: stored, kept_reviewed, empty, failed.
    """
    from sqlalchemy import delete, insert, select
    table = _table()
    stamp = await _read_data_stamp(connection)
    outcome: Dict[str, List[str]] = {"stored": [], "kept_reviewed": [], "empty": [], "failed": []}
    for item in translations:
        question = item["question"]
        key = question_key(question)
        existing = (await connection.execute(
            select(table.c.status).where(table.c.question_key == key))).first()
        if existing is not None and existing.status != UNREVIEWED:
            outcome["kept_reviewed"].append(question)
            continue
        try:
            final_sql, total = await validate(question, item["plan"], item.get("model_sql"))
        except Exception as exc:  # noqa: BLE001
            outcome["failed"].append(f"{question}  [{str(exc).splitlines()[0][:120] if str(exc) else type(exc).__name__}]")
            continue
        if not total:
            outcome["empty"].append(question)
            continue
        await connection.execute(delete(table).where(table.c.question_key == key))
        await connection.execute(insert(table).values(
            question_key=key, question_norm=normalize_question(question), status=UNREVIEWED,
            plan=item["plan"], model_sql=item.get("model_sql"),
            pipeline_fp=pipeline_fingerprint(), vocabulary_fp=vocabulary_fingerprint(),
            schema_fp=schema_fingerprint(), data_stamp=stamp, created_at=_now(),
            final_sql=final_sql, result_total=total, hits=0,
            code_commit=(item.get("code_commit") or PRE_TRACKING)[:64],
        ))
        outcome["stored"].append(question)
    return outcome
