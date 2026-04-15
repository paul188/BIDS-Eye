#!/bin/bash
#SBATCH --job-name=gen-training-data
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=/home/s24pjoha_hpc/Text_To_SQL/training_data_generation/job_%j.log

set -euo pipefail

REPO=/home/s24pjoha_hpc/Text_To_SQL
SCRATCH=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data

module use /opt/software/easybuild-AMD/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load PostgreSQL/16.1-GCCcore-12.3.0
source $REPO/BIDS-SQL/venv_hpc/bin/activate

cd $REPO

# ── PostgreSQL snapshot/startup ────────────────────────────────────────────────
PG_DATA=$SCRATCH/postgres_data
PG_SNAP=$TMPDIR/pg_generation_snap
PG_PORT="${PG_PORT:-5432}"
PG_LOG=$REPO/training_data_generation/pg_generation_${SLURM_JOB_ID}.log

if [[ ! -d "$PG_DATA/base" ]]; then
    echo "[pipeline-job] ERROR: No PostgreSQL data directory at $PG_DATA" >&2
    echo "[pipeline-job] Run a crawl job first so the shared database exists." >&2
    exit 1
fi

# Load API key from file if not already in environment
# Store your key once with: echo "your_key" > ~/.gemini_api_key && chmod 600 ~/.gemini_api_key
if [[ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
    if [[ -f "$HOME/.gemini_api_key" ]]; then
        export GEMINI_API_KEY="$(cat $HOME/.gemini_api_key)"
    else
        echo "Set GEMINI_API_KEY or store it in ~/.gemini_api_key" >&2
        exit 1
    fi
fi

echo "[pipeline-job] Snapshotting PostgreSQL data to $PG_SNAP ..."
rsync -a --exclude='postmaster.pid' --exclude='postmaster.opts' \
    "$PG_DATA/" "$PG_SNAP/"

cleanup() {
    if [[ -d "$PG_SNAP" ]]; then
        pg_ctl -D "$PG_SNAP" stop -m fast >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "[pipeline-job] Starting PostgreSQL on port $PG_PORT ..."
pg_ctl -D "$PG_SNAP" -l "$PG_LOG" -o "-p $PG_PORT" start -w -t 60

if ! psql -h localhost -p "$PG_PORT" -U user -d bids_sql -c "SELECT COUNT(*) FROM bids_datasets;" >/dev/null 2>&1; then
    echo "[pipeline-job] ERROR: PostgreSQL failed to start cleanly. Check $PG_LOG" >&2
    exit 1
fi

export DB_URL="postgresql://user:password@localhost:$PG_PORT/bids_sql"
echo "[pipeline-job] DB ready at $DB_URL"

export N_PROMPTS="${N_PROMPTS:-200}"
export PAIRS_PER_PROMPT="${PAIRS_PER_PROMPT:-24}"
export MODEL="${MODEL:-gemini-3.1-flash-lite-preview}"
export OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-1}"
export OVERWRITE_GOLD="${OVERWRITE_GOLD:-1}"

# ── Gemini API key ─────────────────────────────────────────────────────────────
# Set this before submitting: export GEMINI_API_KEY=... && sbatch ...
# Or hard-code it here (not recommended for shared systems):
# export GEMINI_API_KEY=your_key_here
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

training_data_generation/run_training_data_generation.sh
