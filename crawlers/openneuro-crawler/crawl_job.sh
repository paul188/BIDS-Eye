#!/bin/bash
#SBATCH --job-name=openneuro-crawl
#SBATCH --output=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs/crawl_%j.log
#SBATCH --error=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs/crawl_%j.log
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=intelsr_long

set -e  # exit immediately on any error

# ── Modules ───────────────────────────────────────────────────────────────────
module use /opt/software/easybuild-AMD/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load PostgreSQL/16.1-GCCcore-12.3.0

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/s24pjoha_hpc/Text_To_SQL
VENV=$REPO/BIDS-SQL/venv_hpc
CRAWLER=$REPO/crawlers/openneuro-crawler
SCRATCH=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data
PG_DATA=$SCRATCH/postgres_data
PG_PORT=5429
PG_LOG=$PG_DATA/logfile

export ENV_FILE=$REPO/BIDS-SQL/.env
export BIDS_SQL_CACHE=$SCRATCH/.bids-sql
export CRAWL_DELAY_SECONDS=0.5

# ── Activate venv ─────────────────────────────────────────────────────────────
source $VENV/bin/activate
pip install --quiet apscheduler pyyaml boto3 requests

# ── Prepare directories ───────────────────────────────────────────────────────
mkdir -p $SCRATCH/logs $SCRATCH $BIDS_SQL_CACHE

# ── PostgreSQL: always stop any existing instance, then start clean ───────────
if [ ! -d "$PG_DATA/base" ]; then
    echo "[job] Initialising PostgreSQL cluster at $PG_DATA ..."
    initdb -D $PG_DATA
fi

echo "[job] Ensuring clean PostgreSQL start on $(hostname) ..."

# Stop any running instance (including stale ones from other nodes)
pg_ctl -D $PG_DATA stop -m fast 2>/dev/null && echo "[job] Stopped existing instance." || true

# Remove any leftover lock/pid files
rm -f $PG_DATA/postmaster.pid

# Start fresh and wait until ready
echo "[job] Starting PostgreSQL on port $PG_PORT ..."
pg_ctl -D $PG_DATA -l $PG_LOG -o "-p $PG_PORT" start -w -t 120

# Verify connectivity before proceeding
echo "[job] Verifying PostgreSQL connectivity ..."
for i in $(seq 1 10); do
    psql -h localhost -p $PG_PORT -d postgres -c "SELECT 1;" > /dev/null 2>&1 && break
    echo "[job] Waiting for PostgreSQL... ($i/10)"
    sleep 3
done
psql -h localhost -p $PG_PORT -d postgres -c "SELECT 1;" > /dev/null 2>&1 \
    || { echo "[job] ERROR: PostgreSQL not reachable after start. Check $PG_LOG"; exit 1; }
echo "[job] PostgreSQL is ready."

# ── Create DB / user if they don't exist yet ──────────────────────────────────
psql -h localhost -p $PG_PORT -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='user'" \
    | grep -q 1 || psql -h localhost -p $PG_PORT -d postgres -c "CREATE USER \"user\" WITH PASSWORD 'password';"

psql -h localhost -p $PG_PORT -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='bids_sql'" \
    | grep -q 1 || psql -h localhost -p $PG_PORT -d postgres -c "CREATE DATABASE bids_sql OWNER \"user\";"

# ── Create tables (idempotent) ────────────────────────────────────────────────
echo "[job] Creating DB tables if needed ..."
set +e  # allow python failures here (tables may already exist)
python -c "
import asyncio
from db.db import create_all_tables
asyncio.run(create_all_tables())
"
set -e

# ── Schema migrations ─────────────────────────────────────────────────────────
echo "[job] Applying schema migrations ..."
psql -h localhost -p $PG_PORT -d bids_sql -U user -c "
ALTER TABLE bids_datasets ALTER COLUMN name TYPE VARCHAR(300);
ALTER TABLE bids_datasets ALTER COLUMN bids_version TYPE TEXT;
ALTER TABLE bids_datasets ALTER COLUMN dataset_type TYPE VARCHAR(50);
ALTER TABLE bids_datasets ALTER COLUMN source_type TYPE VARCHAR(50);
ALTER TABLE bids_datasets ALTER COLUMN validation_status TYPE VARCHAR(50);
ALTER TABLE bids_participants ALTER COLUMN sex TYPE VARCHAR(50);
ALTER TABLE bids_participants ALTER COLUMN handedness TYPE VARCHAR(50);
ALTER TABLE bids_objects ALTER COLUMN datatype TYPE TEXT;
ALTER TABLE bids_objects ALTER COLUMN extension TYPE TEXT;
" 2>/dev/null || true

# ── Use curated accessions.yaml (failed retries + not-yet-crawled tail) ──────
# Current queue is maintained manually from the latest crawl log.
# To regenerate from upstream accessions instead: python fetch_accessions.py --out accessions.yaml
cd $CRAWLER
echo "[job] Using accessions.yaml ($(grep -c '^- ds' accessions.yaml) datasets)"

# ── Run one full crawl cycle ───────────────────────────────────────────────────
set +e  # crawler handles its own errors per-dataset
echo "[job] Starting crawl cycle ..."
python -c "
import asyncio
from crawler_service import run_crawl_cycle
asyncio.run(run_crawl_cycle())
"

echo "[job] Done."
