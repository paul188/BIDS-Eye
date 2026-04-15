#!/usr/bin/env python3
"""
expand_labels.py — Robust synonym generation with improved resumption logic.
"""

from __future__ import annotations

import os
import yaml
import time
import json
import argparse
from pathlib import Path
from google import genai
from google.genai import types

# Global flag to track if we have reached the starting point
STARTED = False

SYNONYM_PROMPT_TEMPLATE = """
You are a neuroimaging and behavioral science expert. 
I have a dataset mapping with a specific human-readable label: "{label}"
This label belongs to the category: "{category}"

Please provide a JSON list of 3-5 high-quality synonyms or natural language variations 
that a researcher or clinician might use when searching for this specific concept.

CRITICAL INSTRUCTIONS:
- Must be meaning-preserving and medically/scientifically accurate.
- Mix formal academic terms and common shorthand.
- If the term is highly specific and no common synonyms or variations exist, return an empty list [].
- Return ONLY a JSON array of strings. No prose or markdown fences.
"""

MODEL_CASCADE = [
    # Das aktuelle Flaggschiff für Geschwindigkeit & Intelligenz (GA-Status)
    ("gemini-3.1-pro",         "Primary",    2, [30, 60]),

    # Der hocheffiziente Allrounder (stabil und kostengünstig)
    ("gemini-3.1-flash",       "Fallback 1", 2, [30, 60]),

    # Die Ultra-Leichtgewicht-Variante für einfache Aufgaben/Massendaten
    ("gemini-3.1-flash-lite",  "Fallback 2", 3, [60, 120, 240]),

    # Optional: Spezialmodell für komplexe Logik (Reasoning-Fokus)
    ("gemini-3-deep-think",    "Fallback 3", 5, [120, 240, 480]),
]

_MODEL_IDS = [model_id for model_id, *_ in MODEL_CASCADE]

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, indent=2)


def get_synonyms_with_cascade(client: genai.Client, label: str, category: str, _base_model: str) -> list[str]:
    prompt = SYNONYM_PROMPT_TEMPLATE.format(label=label, category=category)

    model_cascade = list(_MODEL_IDS)

    max_retries_per_model = 5
    base_backoff = 30.0

    print(f"🔍 Starting synonym search for '{label}'...")

    for model_idx, model in enumerate(model_cascade):
        print(f"  [Model {model_idx + 1}/{len(model_cascade)}]: Trying {model}...")
        
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(
                        temperature=0.3, 
                        response_mime_type="application/json"
                    ),
                    contents=prompt
                )
                
                if not response.text:
                    print(f"    ⚠️ Warning: Empty response from {model}. Retrying...")
                    continue
                
                synonyms = json.loads(response.text)
                if isinstance(synonyms, list):
                    print(f"    ✅ Success! Found {len(synonyms)} synonyms using {model}.")
                    return synonyms
                else:
                    print(f"    ⚠️ Warning: Expected list but got {type(synonyms)}. Returning empty.")
                    return []

            except Exception as exc:
                err = str(exc)
                # Group error types
                is_404 = "404" in err or "NOT_FOUND" in err
                is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
                is_overloaded = "503" in err or "UNAVAILABLE" in err
                
                if is_404:
                    print(f"    ❌ Model {model} not found (404). Skipping to next model...")
                    break  # Move to next model in cascade
                
                if is_rate_limit or is_overloaded:
                    if attempt < max_retries_per_model - 1:
                        wait_time = base_backoff * (2 ** attempt)
                        reason = "Rate Limit (429)" if is_rate_limit else "Server Overloaded (503)"
                        print(f"    ⏳ {reason}. Waiting {wait_time}s before attempt {attempt + 2}...")
                        time.sleep(wait_time)
                    else:
                        print(f"    🚫 Max retries reached for {model} due to congestion.")
                        break # Move to next model
                else:
                    # Unexpected errors (Auth, Network, etc.)
                    print(f"    💥 Unexpected error with {model}: {err[:100]}")
                    break # Move to next model

    print(f"  ❌ Failed to retrieve synonyms for '{label}' after exhausting all models.")
    return []

def main():
    global STARTED
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, default="value_mappings.yaml")
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro",
        help="Deprecated compatibility flag; the canonical MODEL_CASCADE is always used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-after", type=str, default="tbi", help="Key to start after")
    args = parser.parse_args()

    if not args.start_after:
        STARTED = True

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set API key env var.")

    client = genai.Client(api_key=api_key)
    data = load_yaml(args.yaml)

    def process_node(node, category_name="general"):
        global STARTED
        if not isinstance(node, dict):
            return

        for k, v in node.items():
            if not STARTED:
                if k == args.start_after:
                    print(f"Target '{k}' found. Processing begins now.")
                    STARTED = True
                if isinstance(v, dict):
                    process_node(v, category_name=k)
                continue

            if isinstance(v, dict) and "label" in v:
                label = v["label"]
                
                # 1. Skip if the label is None or explicitly a filter
                if label is None or str(label).lower() in ["none", "null", "filter"]:
                    continue

                # 2. Skip if already processed
                if not args.overwrite and v.get("synonyms"):
                    continue

                print(f"Generating synonyms for: {label} ({category_name})...")
                syns = get_synonyms_with_cascade(client, label, category_name, args.model)
                
                # 3. Safe list comprehension
                clean_label = str(label).lower().strip()
                v["synonyms"] = [
                    s.lower().strip() for s in syns 
                    if isinstance(s, str) and s.lower().strip() != clean_label
                ]
                print(f"  Added: {v['synonyms']}")
                time.sleep(1.0)

            if isinstance(v, dict):
                process_node(v, category_name=k)

    try:
        # Start at the top level
        for key, content in data.items():
            process_node({key: content})
    finally:
        save_yaml(args.yaml, data)
        print(f"\nProgress saved to {args.yaml}")

if __name__ == "__main__":
    main()
