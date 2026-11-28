"""Smoke test: python -m preprocess "a question" (from backend/llm/)."""
from preprocess import create_query_plan

import sys

test_queries = [
    "Find datasets with at least 15 male ambidextrous participants diagnosed with schizophrenia",
    "Which datasets include fMRI data with an n-back working memory task?",
    "Show me datasets authored by Karl Friston",
    "Find datasets with more female than male participants",
    "Which datasets lack any EEG data but have at least 50 subjects?",
    "List datasets where the same subject has both a T1w scan and did a resting state fMRI",
    "Find datasets where RepetitionTime is greater than 2.0 seconds",
    "Show the top 10 datasets by number of subjects",
    "Datasets funded by the NIH with a CC0 license",
]

query = sys.argv[1] if len(sys.argv) > 1 else test_queries[0]
print(f"Query: {query}\n")

result = create_query_plan(query)
print(result.model_dump_json(indent=2))
