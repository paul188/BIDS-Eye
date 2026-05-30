"""
Synonym collision detector for value_mappings.yaml.

For every synonym string of concept A, compute WRatio against:
  - The label of every other concept B in the same field
  - The high-weight (w >= 0.7) synonyms of every other concept B

Severity:
  CRITICAL  score == 100  (exact string — will always steal queries)
  HIGH      score  95-99  (near-certain steal)
  MEDIUM    score  90-94  (likely steal)

Usage:
    # Fast fuzzy scan of a single field
    python3 RAG/audit_collisions.py --yaml RAG/value_mappings.yaml --field task

    # All fields
    python3 RAG/audit_collisions.py --yaml RAG/value_mappings.yaml

    # Save to file
    python3 RAG/audit_collisions.py --yaml RAG/value_mappings.yaml --out RAG/collision_report.txt

    # Add embedding cosine similarity check (slow, requires sentence-transformers)
    python3 RAG/audit_collisions.py --yaml RAG/value_mappings.yaml --embed --embed-threshold 0.85
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz import process as _process
    import numpy as _np
    _RAPIDFUZZ_OK = True
except ImportError:
    _RAPIDFUZZ_OK = False
    print("ERROR: rapidfuzz not installed. Run: pip install rapidfuzz", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

_CONCEPT_META = frozenset({
    "label", "standard_code", "is_group", "broader", "description",
    "synonyms", "codes", "extra_codes", "dataset_codes", "count",
})


def _extract_synonyms(raw: Any) -> List[Tuple[str, float]]:
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    result: List[Tuple[str, float]] = []
    for s in raw:
        if isinstance(s, str):
            term = s.strip()
            if term:
                result.append((term, 1.0))
        elif isinstance(s, dict) and "term" in s:
            term = str(s["term"]).strip()
            weight = float(s.get("weight", 1.0))
            if term:
                result.append((term, max(0.0, min(1.0, weight))))
    return result


def _load_concepts(path: Path) -> Dict[str, List[dict]]:
    """Load value_mappings.yaml and return {field: [concept_info, ...]}."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    fields: Dict[str, List[dict]] = {}
    for field, concepts in data.items():
        if not isinstance(concepts, dict):
            continue
        concept_list = []
        for key, value in concepts.items():
            if not isinstance(value, dict):
                continue
            label = value.get("label") or key
            synonyms = _extract_synonyms(value.get("synonyms"))
            concept_list.append({
                "key": key,
                "label": label,
                "standard_code": value.get("standard_code"),
                "is_group": bool(value.get("is_group")),
                "synonyms": synonyms,
            })
        if concept_list:
            fields[field] = concept_list
    return fields


# ---------------------------------------------------------------------------
# Fuzzy collision detection
# ---------------------------------------------------------------------------

def _build_targets(concepts: List[dict], high_weight_threshold: float = 0.7) -> Tuple[List[str], List[str]]:
    """Return (label_texts, label_keys) and compile high-weight synonyms separately."""
    label_texts: List[str] = []
    label_keys: List[str] = []
    hw_texts: List[str] = []
    hw_keys: List[str] = []

    for c in concepts:
        label_texts.append(c["label"].lower())
        label_keys.append(c["key"])
        for term, weight in c["synonyms"]:
            if weight >= high_weight_threshold:
                hw_texts.append(term.lower())
                hw_keys.append(c["key"])

    return label_texts, label_keys, hw_texts, hw_keys


_SEVERITY_CRITICAL = 100
_SEVERITY_HIGH = 95
_SEVERITY_MEDIUM = 90


def _classify_severity(score: float) -> Optional[str]:
    if score >= _SEVERITY_CRITICAL:
        return "CRITICAL"
    if score >= _SEVERITY_HIGH:
        return "HIGH"
    if score >= _SEVERITY_MEDIUM:
        return "MEDIUM"
    return None


def _find_fuzzy_collisions(
    concepts: List[dict],
    threshold: int = _SEVERITY_MEDIUM,
    high_weight_threshold: float = 0.7,
) -> List[dict]:
    """
    For each synonym of concept A, compare against:
      - all labels of concepts B (B != A)
      - all high-weight synonyms of concepts B (B != A)

    Returns list of collision dicts.
    """
    if not concepts:
        return []

    label_texts, label_keys, hw_texts, hw_keys = _build_targets(concepts, high_weight_threshold)

    collisions: List[dict] = []
    seen: set = set()  # deduplicate by (key_a, key_b, syn_a, target_text)

    for concept_a in concepts:
        key_a = concept_a["key"]
        synonyms_a = concept_a["synonyms"]
        if not synonyms_a:
            continue

        syn_texts_a = [t.lower() for t, _ in synonyms_a]
        syn_weights_a = [w for _, w in synonyms_a]
        syn_orig_a = [t for t, _ in synonyms_a]

        # Compare synonyms_a vs all labels
        if label_texts:
            scores_labels = _process.cdist(
                syn_texts_a, label_texts, scorer=_fuzz.WRatio, dtype=float
            )
            for i, (score_row) in enumerate(scores_labels):
                for j, score in enumerate(score_row):
                    if score < threshold:
                        continue
                    key_b = label_keys[j]
                    if key_b == key_a:
                        continue
                    dedup_key = (key_a, key_b, syn_texts_a[i], "label", label_texts[j])
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    severity = _classify_severity(score)
                    if severity:
                        collisions.append({
                            "severity": severity,
                            "score": int(score),
                            "source_key": key_a,
                            "source_syn": syn_orig_a[i],
                            "source_syn_weight": syn_weights_a[i],
                            "target_key": key_b,
                            "target_text": label_texts[j],
                            "target_type": "label",
                        })

        # Compare synonyms_a vs high-weight synonyms of other concepts
        if hw_texts:
            scores_hw = _process.cdist(
                syn_texts_a, hw_texts, scorer=_fuzz.WRatio, dtype=float
            )
            for i, score_row in enumerate(scores_hw):
                for j, score in enumerate(score_row):
                    if score < threshold:
                        continue
                    key_b = hw_keys[j]
                    if key_b == key_a:
                        continue
                    dedup_key = (key_a, key_b, syn_texts_a[i], "synonym", hw_texts[j])
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    severity = _classify_severity(score)
                    if severity:
                        collisions.append({
                            "severity": severity,
                            "score": int(score),
                            "source_key": key_a,
                            "source_syn": syn_orig_a[i],
                            "source_syn_weight": syn_weights_a[i],
                            "target_key": key_b,
                            "target_text": hw_texts[j],
                            "target_type": "synonym",
                        })

    # Sort: CRITICAL first, then by score descending
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    collisions.sort(key=lambda c: (severity_order.get(c["severity"], 3), -c["score"]))
    return collisions


# ---------------------------------------------------------------------------
# Embedding collision detection (optional)
# ---------------------------------------------------------------------------

def _find_embedding_collisions(
    concepts: List[dict],
    model_name: str = "FremyCompany/BioLORD-2023-C",
    threshold: float = 0.85,
) -> List[dict]:
    """Embed every concept label, find pairs with cosine >= threshold."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("WARNING: sentence-transformers not installed, skipping embedding check.", file=sys.stderr)
        return []

    print(f"  Loading embedding model {model_name}...")
    model = SentenceTransformer(model_name)
    labels = [c["label"] for c in concepts]
    keys = [c["key"] for c in concepts]

    print(f"  Embedding {len(labels)} concept labels...")
    embeddings = model.encode(labels, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    # Cosine similarity matrix (since embeddings are L2-normalized, dot product = cosine)
    sim_matrix = embeddings @ embeddings.T

    collisions: List[dict] = []
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                collisions.append({
                    "severity": "EMBED",
                    "score": round(score, 3),
                    "key_a": keys[i],
                    "label_a": labels[i],
                    "key_b": keys[j],
                    "label_b": labels[j],
                })

    collisions.sort(key=lambda c: -c["score"])
    return collisions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_fuzzy_collision(c: dict) -> str:
    return (
        f"{c['severity']:<8} "
        f"{c['source_key']}.synonyms[\"{c['source_syn']}\"] (w={c['source_syn_weight']:.1f}, score={c['score']}) "
        f"→ {c['target_key']}.{c['target_type']}[\"{c['target_text']}\"]"
    )


def _format_embed_collision(c: dict) -> str:
    return (
        f"{'EMBED':<8} cosine={c['score']:.3f}  "
        f"{c['key_a']} \"{c['label_a']}\"  ≈  {c['key_b']} \"{c['label_b']}\""
    )


def _print_field_report(
    field: str,
    fuzzy_collisions: List[dict],
    embed_collisions: List[dict],
    n_concepts: int,
    verbose: bool = False,
) -> dict:
    severity_counts = defaultdict(int)
    for c in fuzzy_collisions:
        severity_counts[c["severity"]] += 1

    n_critical = severity_counts["CRITICAL"]
    n_high = severity_counts["HIGH"]
    n_medium = severity_counts["MEDIUM"]
    n_embed = len(embed_collisions)
    n_total_fuzzy = len(fuzzy_collisions)

    print(f"\n{'─'*78}")
    print(f"  Field: {field}   ({n_concepts} concepts, {n_total_fuzzy} fuzzy collisions, {n_embed} embedding collisions)")
    print(f"  CRITICAL={n_critical}  HIGH={n_high}  MEDIUM={n_medium}  EMBED={n_embed}")
    print(f"{'─'*78}")

    if verbose and fuzzy_collisions:
        print("\n  Fuzzy collisions:")
        for c in fuzzy_collisions:
            print("  " + _format_fuzzy_collision(c))

    if verbose and embed_collisions:
        print("\n  Embedding collisions:")
        for c in embed_collisions:
            print("  " + _format_embed_collision(c))

    if not verbose and fuzzy_collisions:
        # Show only CRITICAL and HIGH when not verbose
        important = [c for c in fuzzy_collisions if c["severity"] in ("CRITICAL", "HIGH")]
        if important:
            print("\n  Critical / High fuzzy collisions:")
            for c in important:
                print("  " + _format_fuzzy_collision(c))
        if n_medium > 0:
            print(f"  ({n_medium} MEDIUM collisions — use --verbose to show)")

    return {
        "n_concepts": n_concepts,
        "n_fuzzy": n_total_fuzzy,
        "critical": n_critical,
        "high": n_high,
        "medium": n_medium,
        "embed": n_embed,
    }


def _print_summary_table(summary: Dict[str, dict]) -> None:
    print(f"\n{'='*78}")
    print("  Summary by field")
    print(f"{'='*78}")
    header = f"{'Field':<16} {'Concepts':>9} {'Fuzzy':>7} {'Critical':>9} {'High':>6} {'Medium':>8} {'Embed':>6}"
    print(header)
    print("-" * 78)
    for field, m in sorted(summary.items()):
        print(
            f"{field:<16} {m['n_concepts']:>9} {m['n_fuzzy']:>7} "
            f"{m['critical']:>9} {m['high']:>6} {m['medium']:>8} {m['embed']:>6}"
        )


def _write_report(
    path: Path,
    field_data: Dict[str, Tuple[List[dict], List[dict]]],
    summary: Dict[str, dict],
) -> None:
    lines: List[str] = [
        "# BIDS-Eye synonym collision report",
        f"# Generated by RAG/audit_collisions.py",
        "",
    ]
    for field, (fuzzy_cols, embed_cols) in sorted(field_data.items()):
        m = summary[field]
        lines.append(f"## {field}  ({m['n_concepts']} concepts)")
        lines.append(f"# CRITICAL={m['critical']}  HIGH={m['high']}  MEDIUM={m['medium']}  EMBED={m['embed']}")
        lines.append("")
        for c in fuzzy_cols:
            lines.append(_format_fuzzy_collision(c))
        for c in embed_cols:
            lines.append(_format_embed_collision(c))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect synonym collisions in value_mappings.yaml")
    parser.add_argument("--yaml", default="RAG/value_mappings.yaml", help="Path to value_mappings.yaml")
    parser.add_argument("--field", default=None, help="Only scan this field (e.g. task)")
    parser.add_argument("--threshold", type=int, default=90, help="Minimum WRatio score to flag (default: 90)")
    parser.add_argument("--high-weight", type=float, default=0.7, help="Min weight for synonym-vs-synonym check (default: 0.7)")
    parser.add_argument("--embed", action="store_true", help="Also check embedding cosine similarity")
    parser.add_argument("--embed-threshold", type=float, default=0.85, help="Cosine threshold for embedding collisions (default: 0.85)")
    parser.add_argument("--embed-model", default="FremyCompany/BioLORD-2023-C", help="Embedding model name")
    parser.add_argument("--out", default=None, help="Write full report to this file")
    parser.add_argument("--verbose", action="store_true", help="Print all collisions including MEDIUM")
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {yaml_path}...")
    all_fields = _load_concepts(yaml_path)

    if args.field:
        if args.field not in all_fields:
            print(f"ERROR: field '{args.field}' not found. Available: {sorted(all_fields)}", file=sys.stderr)
            sys.exit(1)
        fields_to_scan = {args.field: all_fields[args.field]}
    else:
        fields_to_scan = all_fields

    total_concepts = sum(len(v) for v in fields_to_scan.values())
    print(f"Scanning {len(fields_to_scan)} fields, {total_concepts} concepts total...")

    field_data: Dict[str, Tuple[List[dict], List[dict]]] = {}
    summary: Dict[str, dict] = {}

    for field, concepts in sorted(fields_to_scan.items()):
        print(f"\n[{field}] {len(concepts)} concepts — running fuzzy cdist...")
        fuzzy_cols = _find_fuzzy_collisions(
            concepts,
            threshold=args.threshold,
            high_weight_threshold=args.high_weight,
        )

        embed_cols: List[dict] = []
        if args.embed:
            print(f"[{field}] running embedding cosine check...")
            embed_cols = _find_embedding_collisions(
                concepts,
                model_name=args.embed_model,
                threshold=args.embed_threshold,
            )

        field_data[field] = (fuzzy_cols, embed_cols)
        m = _print_field_report(field, fuzzy_cols, embed_cols, len(concepts), verbose=args.verbose)
        summary[field] = m

    _print_summary_table(summary)

    if args.out:
        _write_report(Path(args.out), field_data, summary)


if __name__ == "__main__":
    main()
