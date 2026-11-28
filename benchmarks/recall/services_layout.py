"""Let the benchmarks import backend/services by its current paths from any backend.

The harnesses run against older backends too, e.g. the live one inside its
container. Two kinds of older layout are handled:

- a flat services/ (services/text_search.py rather than
  services/fulltext/text_search.py): after use_either_layout(), an import such
  as `from services.fulltext import text_search` loads services/text_search.py;
- a module that has since been renamed (OLD_NAMES): `from services.stores
  import translated_queries` loads services/stores/query_cache.py, or
  services/query_cache.py from a flat backend.

Both names then refer to the same module object, so patching one patches the
other. With a current backend nothing changes.

Call it after the backend folder is on sys.path and before importing services.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys

# Kept in step with backend/services/__init__.py.
GROUPS = {
    "translate": ("text_to_sql", "participant_query_compiler", "sql_rewriter", "concept_keys",
                  "value_check"),
    "fulltext": ("text_search", "tsquery", "ranking", "vectors", "word_frequencies",
                 "word_form_counts"),
    "uncertain": ("tier", "classify", "conditions", "reason_texts", "scanner", "sex", "text"),
    "catalogue": ("catalogue", "schema_capabilities", "catalogue_values"),
    "schema": ("tables", "check"),
    "results": ("rerank",),
    "stores": ("interpretations", "translated_queries"),
    "indexing": ("readme_profile", "dataset_description"),
}

# Current module name -> (its group before the rename, its name before the rename).
OLD_NAMES = {
    "word_form_counts": ("catalogue", "word_form_store"),
    "interpretations": ("stores", "interpretation_store"),
    "translated_queries": ("stores", "query_cache"),
    "tables": ("catalogue", "tables"),
    # check_schema was part of catalogue.py.
    "check": ("catalogue", "catalogue"),
    # The uncertain tier was part of fulltext/, its parts named tier_*.
    "tier": ("fulltext", "uncertain_tier"),
    "classify": ("fulltext", "tier_classify"),
    "conditions": ("fulltext", "tier_conditions"),
    "scanner": ("fulltext", "tier_scanner"),
    "sex": ("fulltext", "tier_sex"),
    "text": ("fulltext", "tier_text"),
}


class _OlderLayout(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds services.<group>[.<module>] under the name an older backend gives it."""

    def __init__(self, flat: bool):
        self._flat = flat
        self._targets: dict[str, str] = {}
        self._own_specs = {}

    def _target(self, group: str, module: str) -> str | None:
        old_group, old_name = OLD_NAMES.get(module, (None, None))
        if self._flat:
            candidates = [f"services.{module}"] + ([f"services.{old_name}"] if old_name else [])
        else:
            candidates = [f"services.{old_group}.{old_name}"] if old_name else []
        return next((c for c in candidates if importlib.util.find_spec(c) is not None), None)

    def find_spec(self, name, path=None, target=None):
        parts = name.split(".")
        if parts[0] != "services" or len(parts) not in (2, 3) or parts[1] not in GROUPS:
            return None
        if len(parts) == 2:
            # The group itself, which this backend lacks (the finder is asked last): an
            # empty package. A flat backend has no group, an older grouped one not all.
            return importlib.util.spec_from_loader(name, self, is_package=True)
        if parts[2] not in GROUPS[parts[1]]:
            return None
        found = self._target(parts[1], parts[2])
        if found is None:
            return None  # a module this backend does not have: a normal ImportError
        self._targets[name] = found
        return importlib.util.spec_from_loader(name, self, is_package=False)

    def create_module(self, spec):
        found = self._targets.pop(spec.name, None)
        if found is None:
            return None
        module = importlib.import_module(found)
        self._own_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        # The import system sets __spec__ to the alias's spec; give the module its own back.
        own = self._own_specs.pop(getattr(module.__spec__, "name", None), None)
        if own is not None:
            module.__spec__ = own


def use_either_layout() -> bool:
    """Install the finder for older layouts; True if the backend on sys.path is flat."""
    if any(isinstance(f, _OlderLayout) for f in sys.meta_path):
        return any(isinstance(f, _OlderLayout) and f._flat for f in sys.meta_path)
    if importlib.util.find_spec("services") is None:
        return False
    flat = importlib.util.find_spec("services.translate") is None
    sys.meta_path.append(_OlderLayout(flat))
    return flat


def set_setting(name: str, value: str) -> None:
    """Set a backend setting while the harness runs, e.g. set_setting("TEXT_SEARCH_EXACT", "true").

    An older backend reads the environment at every call. A current one reads
    it once at import into config.settings.settings_obj, which is changed here.
    """
    os.environ[name] = value
    settings = sys.modules.get("config.settings")
    if settings is not None:
        setattr(settings.settings_obj, name, getattr(settings.Settings(), name))
