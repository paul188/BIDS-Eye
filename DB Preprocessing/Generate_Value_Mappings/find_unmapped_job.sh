#!/bin/bash
#SBATCH --job-name=find-unmapped
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:30:00
#SBATCH --output=/home/s24pjoha_hpc/Text_To_SQL/training_data_generation/find_unmapped_%j.log

# Usage:
#   sbatch training_data_generation/find_unmapped_job.sh

set -euo pipefail

REPO=/home/s24pjoha_hpc/Text_To_SQL
SCRATCH=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data
PG_DATA_LUSTRE=$SCRATCH/postgres_data
PG_DATA=/home/s24pjoha_hpc/pg_unmapped_$$   # working copy on home (lower latency)
PG_PORT=5429
PG_LOG=$PG_DATA/logfile
DB_URL="postgresql://user:password@localhost:${PG_PORT}/bids_sql"
OUT=$REPO/training_data_generation/unmapped_db_values.yaml

module use /opt/software/easybuild-AMD/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load PostgreSQL/16.1-GCCcore-12.3.0
source $REPO/BIDS-SQL/venv_hpc/bin/activate

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

# ── Run find_unmapped_db_values ───────────────────────────────────────────────
echo "[unmapped] Scanning for unmapped DB values ..."
python training_data_generation/find_unmapped_db_values.py \
    --db-url "$DB_URL" \
    --out    "$OUT"

echo "[unmapped] Done. Results written to $OUT"

# ── Shutdown and clean up (read-only — no sync back to Lustre) ────────────────
echo "[pg] Stopping PostgreSQL ..."
pg_ctl -D "$PG_DATA" stop -m fast

echo "[pg] Removing working copy from home ..."
rm -rf "$PG_DATA"

echo "[done]"
