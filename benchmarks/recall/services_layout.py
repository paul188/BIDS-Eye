"""Let the benchmarks import backend/services by its grouped paths from any backend.

The harnesses run against older backends too, e.g. the live one inside its
container, whose services/ is still flat (services/text_search.py rather than
services/fulltext/text_search.py). After use_either_layout(), an import such
as `from services.fulltext import text_search` loads services/text_search.py
from such a backend; with a grouped backend it changes nothing. Both names
then refer to the same module object, so patching one patches the other.

Call it after the backend folder is on sys.path and before importing services.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

# Kept in step with backend/services/__init__.py.
GROUPS = {
    "translate": ("text_to_sql", "participant_query_compiler", "sql_rewriter", "concept_keys",
                  "value_check"),
    "fulltext": ("text_search", "tsquery", "ranking", "vectors", "word_frequencies",
                 "uncertain_tier", "tier_classify", "tier_conditions", "tier_scanner", "tier_sex",
                 "tier_text"),
    "catalogue": ("schema_capabilities", "catalogue_values", "word_form_store"),
    "results": ("query_filters", "rerank", "dataset_cards", "readme_snippets"),
    "stores": ("interpretation_store", "query_cache"),
    "indexing": ("readme_profile", "dataset_description"),
}


class _FlatLayout(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds services.<group>[.<module>] in a flat services/ folder."""

    def find_spec(self, name, path=None, target=None):
        parts = name.split(".")
        if parts[0] != "services" or len(parts) not in (2, 3) or parts[1] not in GROUPS:
            return None
        if len(parts) == 3 and (parts[2] not in GROUPS[parts[1]]
                                or importlib.util.find_spec(f"services.{parts[2]}") is None):
            return None  # a module this backend does not have: a normal ImportError
        return importlib.util.spec_from_loader(name, self, is_package=len(parts) == 2)

    def __init__(self):
        self._own_specs = {}

    def create_module(self, spec):
        parts = spec.name.split(".")
        if len(parts) == 3:
            module = importlib.import_module(f"services.{parts[2]}")
            self._own_specs[spec.name] = module.__spec__
            return module
        return None  # the group itself: an empty package

    def exec_module(self, module):
        # The import system sets __spec__ to the alias's spec; give the module its own back.
        own = self._own_specs.pop(getattr(module.__spec__, "name", None), None)
        if own is not None:
            module.__spec__ = own


def use_either_layout() -> bool:
    """Install the flat-layout finder if the backend on sys.path needs it; True if it did."""
    if any(isinstance(f, _FlatLayout) for f in sys.meta_path):
        return True
    if importlib.util.find_spec("services") is None:
        return False
    if importlib.util.find_spec("services.translate") is not None:
        return False  # grouped layout: nothing to do
    sys.meta_path.append(_FlatLayout())
    return True
