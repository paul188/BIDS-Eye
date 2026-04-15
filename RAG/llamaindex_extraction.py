from pydantic import BaseModel, Field
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core.program import LLMTextCompletionProgram

from yaml_to_llamaindex import group_db, get_group_summary
from llama_index.llms.google_genai import GoogleGenAI


# ---------------------------------------------------------------------------
# 1. Schema for structured LLM output
# ---------------------------------------------------------------------------

class UserQueryEntities(BaseModel):
    diagnosis: List[str] = Field(
        default_factory=list,
        description=(
            "Clinical conditions, health status, or diagnostic groups. "
            "Include both specific diagnoses ('major depressive disorder', 'focal epilepsy') "
            "AND broad group terms ('psychiatric disorders', 'mood disorders', "
            "'neurodevelopmental', 'healthy controls') when mentioned."
        ),
    )
    task: List[str] = Field(
        default_factory=list,
        description=(
            "Cognitive or behavioural tasks performed during data acquisition. "
            "Include specific task names ('n-back', 'stroop') AND broad categories "
            "('working memory', 'attention', 'resting state') when mentioned."
        ),
    )
    suffix: List[str] = Field(
        default_factory=list,
        description="File suffixes or imaging modalities, e.g. 'bold', 'T1w', 'eeg'.",
    )
    handedness: List[str] = Field(
        default_factory=list,
        description="Dominant hand of participants, e.g. 'left-handed', 'right-handed'.",
    )
    sex: List[str] = Field(
        default_factory=list,
        description="Biological sex of participants, e.g. 'male', 'female'.",
    )
    datatype: List[str] = Field(
        default_factory=list,
        description="Data modality folders, e.g. 'func', 'anat', 'eeg', 'dwi'.",
    )
    sidecar_fields: List[str] = Field(
        default_factory=list,
        description="JSON sidecar metadata fields referenced in the query.",
    )
    participant_extra_fields: List[str] = Field(
        default_factory=list,
        description="Additional participant-level variables mentioned in the query.",
    )


# ---------------------------------------------------------------------------
# 2. Build a compact group-vocabulary hint for the prompt
# ---------------------------------------------------------------------------

def _build_group_hint() -> str:
    """Create a brief taxonomy summary to orient the LLM."""
    summary = get_group_summary(group_db)
    lines = []
    for cat, terms in summary.items():
        # Show a sample of group-level terms so the prompt stays concise
        sample = terms[:12]
        sample_str = ", ".join(f'"{t}"' for t in sample)
        if len(terms) > 12:
            sample_str += f" … ({len(terms)} total group terms)"
        lines.append(f"  {cat}: {sample_str}")
    return "\n".join(lines)


_GROUP_HINT = _build_group_hint()


# ---------------------------------------------------------------------------
# 3. Extraction prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
Extract entities from the user's query into the specified categories.
If a category is not mentioned, leave the list empty.

STRICT RULES:
1. Extract ONLY what is explicitly stated in the query.
   Do NOT infer or guess entities from context clues.
   Example: a query mentioning "working memory" does NOT imply any diagnosis —
   extract a diagnosis only if the query explicitly names one.

2. The downstream resolver understands BOTH specific terms AND broad group/category terms:
   • Specific: "focal epilepsy" → diagnosis,  "n-back" → task
   • Broad group: "psychiatric disorders" → diagnosis,  "memory tasks" → task
   • Both: "schizophrenia and other psychiatric conditions"
           → diagnosis: ["schizophrenia", "psychiatric disorders"]

3. Use the known group vocabulary below to decide whether a broad term should be
   extracted as-is (group expansion will happen downstream).

Known group-level vocabulary (sample, not exhaustive):
{group_hint}

User Query: {{query_str}}
"""

prompt_template_str = _PROMPT_TEMPLATE.format(group_hint=_GROUP_HINT)


# ---------------------------------------------------------------------------
# 4. LLM + extraction program
# ---------------------------------------------------------------------------

llm = GoogleGenAI(model="models/gemini-2.5-flash", temperature = 0.0)

extractor_program = LLMTextCompletionProgram.from_defaults(
    output_cls=UserQueryEntities,
    llm=llm,
    prompt_template_str=prompt_template_str,
)
