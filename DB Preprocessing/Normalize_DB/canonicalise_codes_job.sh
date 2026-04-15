#!/bin/bash
#SBATCH --job-name=canonicalise-codes
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=5:30:00
#SBATCH --output=/home/s24pjoha_hpc/Text_To_SQL/training_data_generation/canonicalise_%j.log

# Usage:
#   sbatch training_data_generation/canonicalise_codes_job.sh           # apply changes
#   sbatch training_data_generation/canonicalise_codes_job.sh --dry-run # preview only

set -euo pipefail

# ── Cancel any currently running canonicalise job before starting ─────────────
SELF=$(basename "$0")
RUNNING=$(squeue --me --name=canonicalise-codes --noheader --format="%i" 2>/dev/null | grep -v "^${SLURM_JOB_ID}$" || true)
if [[ -n "$RUNNING" ]]; then
    echo "[slurm] Cancelling existing canonicalise-codes job(s): $RUNNING"
    scancel $RUNNING
    sleep 5  # give SLURM a moment to clean up
fi

REPO=/home/s24pjoha_hpc/Text_To_SQL
SCRATCH=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data
PG_DATA_LUSTRE=$SCRATCH/postgres_data          # authoritative copy on Lustre
PG_DATA=/home/s24pjoha_hpc/pg_canonicalise_$$ # working copy on home (IB-NFS, lower latency)
PG_PORT=5429
PG_LOG=$PG_DATA/logfile
DB_URL="postgresql://user:password@localhost:${PG_PORT}/bids_sql"
DRY_RUN="${1:-}"   # pass --dry-run as first argument to preview without writing

module use /opt/software/easybuild-AMD/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load PostgreSQL/16.1-GCCcore-12.3.0
source $REPO/BIDS-SQL/venv_hpc/bin/activate

cd $REPO

# ── Copy DB from Lustre to home (InfiniBand NFS, lower random-I/O latency) ───
echo "[pg] Copying DB from Lustre to home (~3 GB) ..."
cp -a "$PG_DATA_LUSTRE" "$PG_DATA"
echo "[pg] Copy done: $PG_DATA"

# Remove any stale pid copied from Lustre
rm -f "$PG_DATA/postmaster.pid"

# Kill any postgres still pointing at the Lustre path from a prior job
pkill -f "postgres.*${PG_DATA_LUSTRE}" 2>/dev/null || true

# ── Start PostgreSQL from home ─────────────────────────────────────────────────
echo "[pg] Starting PostgreSQL on port $PG_PORT ..."
pg_ctl -D "$PG_DATA" -l "$PG_LOG" -o "-p $PG_PORT" start -w -t 120

# Verify
psql -h localhost -p "$PG_PORT" -U user -d bids_sql -c "SELECT COUNT(*) FROM bids_datasets;" > /dev/null
echo "[pg] PostgreSQL ready (running from home)."

# ── Run canonicalisation ──────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[canonicalise] DRY RUN — no changes will be written."
else
    echo "[canonicalise] Applying standard codes to database ..."
fi

python training_data_generation/apply_canonical_codes.py \
    --db-url "$DB_URL" \
    $DRY_RUN

# ── Shutdown ──────────────────────────────────────────────────────────────────
echo "[pg] Stopping PostgreSQL ..."
pg_ctl -D "$PG_DATA" stop -m fast

# ── Sync updated DB back to Lustre (skip for dry-run) ─────────────────────────
if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo "[pg] Syncing updated DB back to Lustre ..."
    rsync -a --delete "$PG_DATA/" "$PG_DATA_LUSTRE/"
    echo "[pg] Lustre copy updated."
fi

# ── Clean up home copy ────────────────────────────────────────────────────────
echo "[pg] Removing working copy from home ..."
rm -rf "$PG_DATA"

echo "[done]"
