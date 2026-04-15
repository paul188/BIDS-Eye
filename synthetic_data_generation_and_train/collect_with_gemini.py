#!/usr/bin/env python3
"""
collect_with_gemini.py — Send prompt files to Gemini and save raw results.

Authentication:
  export GEMINI_API_KEY=...
or
  export GOOGLE_API_KEY=...

Requires:
  pip install google-genai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import SYSTEM

SYSTEM_INSTRUCTION = (
    SYSTEM
    + "\n\n"
    "Additional instructions for data generation:\n"
    "- Follow the prompt exactly.\n"
    "- Return only the requested JSON array — no markdown fences, no prose.\n"
    "- Keep paraphrases meaning-preserving.\n"
    "- Avoid repeating the same SQL skeleton more than 3 times per batch.\n"
    "- Every generated SQL MUST use the required SELECT columns shown in the system prompt above."
)


def iter_prompt_files(prompt_dir: Path) -> Iterable[Path]:
    return sorted(prompt_dir.glob("prompt_*.txt"))


def result_path_for(prompt_path: Path, out_dir: Path) -> Path:
    match = re.search(r"(\d+)", prompt_path.stem)
    idx = match.group(1) if match else prompt_path.stem
    return out_dir / f"result_{idx}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--max-retries", type=int, default=20,
                        help="Max retry attempts per prompt on rate-limit / transient errors")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for prompt_path in iter_prompt_files(args.prompt_dir):
        out_path = result_path_for(prompt_path, args.out_dir)
        if out_path.exists() and not args.overwrite:
            print(f"Skip {prompt_path.name} -> {out_path.name} (already exists)")
            continue

        prompt_text = prompt_path.read_text(encoding="utf-8")
        print(f"Sending {prompt_path.name} to {args.model} ...")

        # JSON response schema — enforces valid structure without relying on
        # prompting alone.  Gemini returns a guaranteed-parseable JSON array.
        _pair_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "question":            types.Schema(type=types.Type.STRING),
                "sql":                 types.Schema(type=types.Type.STRING),
                "pattern":             types.Schema(type=types.Type.STRING),
                "family":              types.Schema(type=types.Type.STRING),
                "sql_structure":       types.Schema(type=types.Type.STRING),
                "paraphrase_bundle_id":types.Schema(
                                           type=types.Type.STRING,
                                           nullable=True,
                                       ),
            },
            required=["question", "sql", "pattern", "family", "sql_structure"],
        )
        _output_schema = types.Schema(
            type=types.Type.ARRAY,
            items=_pair_schema,
        )

        response = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        # Lower temperature improves rule adherence on complex
                        # prompts (15+ constraints); 0.7 keeps enough variety.
                        temperature=0.7,
                        top_p=0.95,
                        # 18 pairs × ~400 tokens each + overhead; give headroom
                        # so the array is never truncated mid-element.
                        max_output_tokens=16384,
                        response_mime_type="application/json",
                        response_schema=_output_schema,
                    ),
                    contents=prompt_text,
                )
                break  # success
            except Exception as exc:
                err = str(exc)
                is_fatal = "404" in err or "NOT_FOUND" in err or "INVALID_ARGUMENT" in err
                if is_fatal:
                    raise RuntimeError(f"{args.model} returned a fatal error: {err}") from exc
                print(f"  Attempt {attempt}/{args.max_retries} failed: {err[:120]}")
                if attempt < args.max_retries:
                    print(f"  Waiting 30 s before retry ...")
                    time.sleep(30)
                else:
                    raise RuntimeError(
                        f"Giving up on {prompt_path.name} after {args.max_retries} attempts"
                    ) from exc

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError(f"No text returned for {prompt_path.name}")
        # Quick structural check: must be a non-empty JSON array of objects.
        # This catches model refusals and schema mismatches before writing.
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list) or len(parsed) == 0:
                raise ValueError("Expected a non-empty JSON array")
            missing = [i for i, p in enumerate(parsed) if "question" not in p or "sql" not in p]
            if missing:
                raise ValueError(f"Elements {missing} missing question or sql")
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  WARN: response for {prompt_path.name} failed validation: {exc} — skipping")
            continue
        out_path.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"Wrote {out_path} ({len(parsed)} pairs)")
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
