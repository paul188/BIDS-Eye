#!/bin/bash
#SBATCH --job-name=build-metadata-index
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home/s24pjoha_hpc/Text_To_SQL/training_data_generation/build_metadata_index_%j.log

# Builds the vector metadata index (metadata_index.json) used by the
# LLM_preprocessor RAG retriever.
#
# Usage:
#   sbatch training_data_generation/build_metadata_index_job.sh
#
# The index is written to $SCRATCH/metadata_index.json and also copied
# to $REPO/LLM_preprocessor/metadata_index.json for local use.

set -euo pipefail

REPO=/home/s24pjoha_hpc/Text_To_SQL
SCRATCH=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data
PG_DATA_LUSTRE=$SCRATCH/postgres_data
PG_DATA=/home/s24pjoha_hpc/pg_metadata_$$   # working copy on home (lower latency)
PG_PORT=5429
PG_LOG=$PG_DATA/logfile
DB_URL="postgresql://user:password@localhost:${PG_PORT}/bids_sql"
INDEX_OUT=$SCRATCH/metadata_index.json
INDEX_LOCAL=$REPO/LLM_preprocessor/metadata_index.json
VENV_FALLBACK=$SCRATCH/train_venv
VENV_PRIMARY=$REPO/BIDS-SQL/venv_hpc

module use /opt/software/easybuild-AMD/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load PostgreSQL/16.1-GCCcore-12.3.0

check_venv() {
    local venv_path=$1
    local label=$2

    if [ ! -d "$venv_path" ]; then
        echo "[env] $label missing: $venv_path"
        return 1
    fi

    local status
    status=$(source "$venv_path/bin/activate" && python - <<'PY'
import importlib.util

required = {
    "sentence_transformers": "sentence-transformers",
    "sqlalchemy": "sqlalchemy",
    "psycopg2": "psycopg2-binary",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("missing: " + ", ".join(missing))
    raise SystemExit(1)
print("ok")
PY
    ) || true

    if [ "$status" = "ok" ]; then
        source "$venv_path/bin/activate"
        echo "[env] Activated $venv_path"
        return 0
    fi

    echo "[env] Skipping unusable $label: $venv_path"
    if [ -n "$status" ]; then
        echo "[env]   $status"
    fi
    return 1
}

if ! check_venv "$VENV_PRIMARY" "primary venv"; then
    if ! check_venv "$VENV_FALLBACK" "fallback venv"; then
        echo "[env] ERROR: no usable venv found."
        echo "[env] Checked:"
        echo "[env]   $VENV_PRIMARY"
        echo "[env]   $VENV_FALLBACK"
        echo "[env] Recreate / update the venv first, for example:"
        echo "[env]   sbatch training/create_train_venv.sh"
        exit 1
    fi
fi

python - <<'PY'
import importlib.util
import sys

required = {
    "sentence_transformers": "sentence-transformers",
    "sqlalchemy": "sqlalchemy",
    "psycopg2": "psycopg2-binary",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    names = ", ".join(missing)
    raise SystemExit(
        "[env] ERROR: active venv is missing required packages: "
        f"{names}\n"
        "[env] Recreate / update the venv first, for example:\n"
        "[env]   sbatch training/create_train_venv.sh"
    )
PY

cd $REPO

# ── Copy DB from Lustre to home ───────────────────────────────────────────────
echo "[pg] Copying DB from Lustre to home ..."
cp -a "$PG_DATA_LUSTRE" "$PG_DATA"
echo "[pg] Copy done: $PG_DATA"

rm -f "$PG_DATA/postmaster.pid"
pkill -f "postgres.*${PG_DATA_LUSTRE}" 2>/dev/null || true

# ── Start PostgreSQL ──────────────────────────────────────────────────────────
echo "[pg] Starting PostgreSQL on port $PG_PORT ..."
pg_ctl -D "$PG_DATA" -l "$PG_LOG" -o "-p $PG_PORT" start -w -t 120

psql -h localhost -p "$PG_PORT" -U user -d bids_sql -c "SELECT COUNT(*) FROM bids_datasets;" > /dev/null
echo "[pg] PostgreSQL ready."

# ── Build index ───────────────────────────────────────────────────────────────
echo "[index] Building metadata index ..."
python modal_app/build_metadata_index.py \
    --db-url "$DB_URL" \
    --out    "$INDEX_OUT"

echo "[index] Written to $INDEX_OUT"

# ── Copy to repo for direct use by the pipeline script ───────────────────────
cp "$INDEX_OUT" "$INDEX_LOCAL"
echo "[index] Copied to $INDEX_LOCAL"

# ── Shutdown and clean up (read-only — no sync back to Lustre needed) ─────────
echo "[pg] Stopping PostgreSQL ..."
pg_ctl -D "$PG_DATA" stop -m fast

echo "[pg] Removing working copy from home ..."
rm -rf "$PG_DATA"

echo "[done] Index available at:"
echo "  $INDEX_OUT"
echo "  $INDEX_LOCAL"
