#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$ROOT_DIR/training_data_generation"

DB_URL="${DB_URL:-}"
PROMPT_DIR="${PROMPT_DIR:-$PIPELINE_DIR/prompts}"
RESULT_DIR="${RESULT_DIR:-$PIPELINE_DIR/results}"
REPORT_DIR="${REPORT_DIR:-$PIPELINE_DIR/reports}"
GOLD_OUT="${GOLD_OUT:-$ROOT_DIR/training.generated.jsonl}"
REPAIR_OUT="${REPAIR_OUT:-$ROOT_DIR/training.needs_repair.jsonl}"
MODEL="${MODEL:-gemini-2.5-flash}"
N_PROMPTS="${N_PROMPTS:-20}"
PAIRS_PER_PROMPT="${PAIRS_PER_PROMPT:-18}"
SEED="${SEED:-42}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1.0}"
OVERWRITE_GOLD="${OVERWRITE_GOLD:-0}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-0}"
ALLOW_MAIN_DATASET_OVERWRITE=0

usage() {
    cat <<EOF
Usage:
  DB_URL="postgresql://user:password@localhost:5429/bids_sql" \\
  GEMINI_API_KEY=... \\
  training_data_generation/run_training_data_generation.sh

Optional environment variables:
  PROMPT_DIR
  RESULT_DIR
  REPORT_DIR
  GOLD_OUT
  REPAIR_OUT
  MODEL
  N_PROMPTS
  PAIRS_PER_PROMPT
  SEED
  SLEEP_SECONDS
  OVERWRITE_GOLD=1
  OVERWRITE_RESULTS=1
  --allow-main-dataset-overwrite

You can also skip stages:
  training_data_generation/run_training_data_generation.sh --skip-sample
  training_data_generation/run_training_data_generation.sh --skip-collect
  training_data_generation/run_training_data_generation.sh --skip-post
EOF
}

SKIP_SAMPLE=0
SKIP_COLLECT=0
SKIP_POST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sample)
            SKIP_SAMPLE=1
            shift
            ;;
        --skip-collect)
            SKIP_COLLECT=1
            shift
            ;;
        --skip-post)
            SKIP_POST=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --allow-main-dataset-overwrite)
            ALLOW_MAIN_DATASET_OVERWRITE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$DB_URL" ]]; then
    echo "DB_URL is required." >&2
    usage
    exit 1
fi

GOLD_BASENAME="$(basename "$GOLD_OUT")"
if [[ "$GOLD_BASENAME" == "training.jsonl" && "$ALLOW_MAIN_DATASET_OVERWRITE" -ne 1 ]]; then
    echo "Refusing to write directly to training.jsonl without --allow-main-dataset-overwrite." >&2
    echo "Set GOLD_OUT to a separate file such as training.generated.jsonl, or pass the explicit override flag." >&2
    exit 1
fi

if [[ "$SKIP_SAMPLE" -eq 0 ]]; then
    echo "[pipeline] Sampling diverse prompts..."
    python "$PIPELINE_DIR/sample_diverse_prompts.py" \
        --db-url "$DB_URL" \
        --out-dir "$PROMPT_DIR" \
        --n-prompts "$N_PROMPTS" \
        --pairs-per-prompt "$PAIRS_PER_PROMPT" \
        --seed "$SEED"
fi

if [[ "$SKIP_COLLECT" -eq 0 ]]; then
    if [[ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
        echo "Set GEMINI_API_KEY or GOOGLE_API_KEY before running the Gemini collection step." >&2
        exit 1
    fi

    echo "[pipeline] Collecting raw Gemini outputs..."
    COLLECT_ARGS=(
        python "$PIPELINE_DIR/collect_with_gemini.py"
        --prompt-dir "$PROMPT_DIR"
        --out-dir "$RESULT_DIR"
        --model "$MODEL"
        --sleep-seconds "$SLEEP_SECONDS"
    )
    if [[ "$OVERWRITE_RESULTS" == "1" ]]; then
        COLLECT_ARGS+=(--overwrite)
    fi
    "${COLLECT_ARGS[@]}"
fi

if [[ "$SKIP_POST" -eq 0 ]]; then
    echo "[pipeline] Validating, merging, and auditing generated data..."
    POST_ARGS=(
        python "$PIPELINE_DIR/post_process_pipeline.py"
        --result-dir "$RESULT_DIR"
        --db-url "$DB_URL"
        --gold-out "$GOLD_OUT"
        --repair-out "$REPAIR_OUT"
        --report-dir "$REPORT_DIR"
    )
    if [[ "$OVERWRITE_GOLD" == "1" ]]; then
        POST_ARGS+=(--overwrite-gold)
    fi
    "${POST_ARGS[@]}"
fi

echo "[pipeline] Done."
echo "[pipeline] Prompts: $PROMPT_DIR"
echo "[pipeline] Results: $RESULT_DIR"
echo "[pipeline] Gold: $GOLD_OUT"
echo "[pipeline] Repair: $REPAIR_OUT"
echo "[pipeline] Reports: $REPORT_DIR"
