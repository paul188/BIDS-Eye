"""
backend/llm/preprocess/
----------------------------
Stage 1 of the Text-to-SQL pipeline.

Flow:
  User query
      │
      ▼
  create_query_plan()          ← intent.py
      │  returns QueryPlan (structured Pydantic object)
      │
      ▼
  RAG  (backend/vocabulary/retriever.py MetadataRetriever)
      │  receives rag_requests from QueryPlan
      │  returns resolved canonical codes per term
      │  scan terms are searched across datatype + suffix + task simultaneously
      │
      ▼
  augment_with_rag()          ← augment.py
      │  returns AugmentedQueryPlan with canonical codes filled in
      │
      ▼
  SQL generation LLM (profile sql, in backend/services/translate/text_to_sql.py)

The LLM's job is STRUCTURAL ONLY:
  - Identify which terms are scan-related (put in ScanRequirement.scan_terms)
  - Identify participant filters (diagnosis, sex, age, handedness)
  - Identify dataset-level filters (author, funding, license, DOI)
  - Identify sidecar / JSONB filters (RepetitionTime > 2.0)

The RAG layer handles ALL cardinality decisions using threshold-based fuzzy
matching.  The LLM never sets expected code counts or classifies scan terms
into datatype vs suffix vs task.

Models and keys: profile intent in backend/llm/llm_config.yaml (llm_client.py);
the env var that holds each provider's key is named there.

Modules, in import order (each uses only those before it):
  plan.py      The plan the model returns (QueryPlan) and the plan after RAG (AugmentedQueryPlan).
  prompt.py    The system prompt of the intent-extraction call.
  intent.py    The model call: a question in, a QueryPlan out (create_query_plan).
  rag.py       What to look up in the vocabulary for a plan, and the scored filters that come back.
  topics.py    Topics: a qualified subject, searched as one unit; description alternatives.
  augment.py   The plan with RAG results: filter lines and the augmented question (augment_with_rag).
  pipeline.py  The full pipeline: question → QueryPlan → RAG → AugmentedQueryPlan (run_pipeline).
  catalogue.py the backend lookups (full-text counts, word forms); ImportError outside the backend
"""
from __future__ import annotations

# The names other code and the tests use (private ones included).
from .plan import (  # noqa: F401
    AugmentedQueryPlan,
    MetadataFilter,
    Ordering,
    ParticipantGroup,
    QueryPlan,
    RAGField,
    RAGRequest,
    ResolvedTerms,
    ResolvedTopic,
    ScanRequirement,
    ScoredFilter,
    SortKey,
    TermAlternatives,
    Topic,
)
from .prompt import (  # noqa: F401
    _SYSTEM_PROMPT,
)
from .intent import (  # noqa: F401
    _plan_output_schema,
    create_query_plan,
    count_stem_forms,
)
from .rag import (  # noqa: F401
    _is_healthy_term,
    _participant_conditions,
    build_rag_requests,
)
from .topics import (  # noqa: F401
    _detach_topic_terms,
    _keep_ambiguity_topics,
    check_description_alternatives,
    check_scan_term_alternatives,
    resolve_topic,
    resolve_topics,
)
from .augment import (  # noqa: F401
    _age_groups_only,
    _build_augmented_question,
    _content_words,
    _diagnosis_lines,
    _formula_matches_too_few,
    _matches_too_few,
    _without_phrases,
    augment_with_rag,
)
from .pipeline import (  # noqa: F401
    _names_that_are_conditions,
    run_pipeline,
)
