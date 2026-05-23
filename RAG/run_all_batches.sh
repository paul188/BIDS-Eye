#!/usr/bin/env bash
# run_all_batches.sh — run synonym expansion for all batches and merge each one
# Usage: bash RAG/run_all_batches.sh
set -euo pipefail

DB_URL="postgresql://bids:changeme@localhost:5432/bids_sql"
YAML="RAG/value_mappings.yaml"
MODEL="gemini-3.5-flash"
BATCH_SIZE=200
TOTAL_NODES=1390

export GEMINI_API_KEY=$(grep GEMINI_API_KEY .env | cut -d= -f2)

ensure_db() {
    if ! psql "$DB_URL" -c "SELECT 1" -q >/dev/null 2>&1; then
        echo "[db] Container not responding — restarting..."
        docker start bids-eye-db-1 >/dev/null 2>&1 || true
        for i in $(seq 1 15); do
            sleep 2
            psql "$DB_URL" -c "SELECT 1" -q >/dev/null 2>&1 && echo "[db] Ready." && return
        done
        echo "[db] ERROR: could not connect after restart. Aborting." >&2
        exit 1
    fi
}

run_batch() {
    local offset=$1
    local size=$2
    local out=$3
    local log="/tmp/batch_${offset}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch offset=$offset  size=$size  → $out"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    ensure_db

    # Retry the whole batch up to 3 times (handles transient DB loss mid-run)
    for attempt in 1 2 3; do
        python3 RAG/expand_synonyms.py \
            --db-url "$DB_URL" \
            --yaml   "$YAML" \
            --mode   replace \
            --max-nodes "$size" \
            --offset    "$offset" \
            --out       "$out" \
            --gemini-model "$MODEL" \
            2>&1 | tee "$log"

        saved=$(grep -c "^- path:" "$out" 2>/dev/null || echo 0)
        if [ "$saved" -ge "$((size / 2))" ]; then
            echo "[batch] $saved/$size proposals saved."
            break
        else
            echo "[batch] Only $saved/$size saved (attempt $attempt/3). Checking DB..."
            ensure_db
        fi
    done
}

merge() {
    local proposals=$1
    local count
    count=$(grep -c "^- path:" "$proposals" 2>/dev/null || echo 0)
    if [ "$count" -eq 0 ]; then
        echo "[merge] $proposals is empty — skipping."
        return
    fi
    echo "[merge] Applying $count proposals from $proposals..."
    python3 RAG/merge_proposals.py \
        --proposals "$proposals" \
        --yaml      "$YAML" \
        --mode      replace
}

# ── Batches 1-4 already complete — merge batch 4 then continue ──────────────
echo "=== Merging completed batch 4 ==="
merge RAG/proposals_batch4.yaml

# ── Remaining batches ───────────────────────────────────────────────────────
offset=800
batch_num=5
while [ "$offset" -lt "$TOTAL_NODES" ]; do
    size=$BATCH_SIZE
    if [ $((offset + size)) -gt "$TOTAL_NODES" ]; then
        size=$((TOTAL_NODES - offset))
    fi

    out="RAG/proposals_batch${batch_num}.yaml"
    run_batch "$offset" "$size" "$out"
    merge "$out"

    offset=$((offset + size))
    batch_num=$((batch_num + 1))
done

echo ""
echo "✓ All batches complete. value_mappings.yaml updated."
echo "  Run: python3 RAG/audit_yaml.py RAG/value_mappings.yaml"
