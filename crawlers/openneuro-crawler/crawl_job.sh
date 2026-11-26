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
pip install --quiet apscheduler pyyaml boto3 requests "google-genai==2.8.0"  # google-genai: post-crawl dataset descriptions

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

# ── Tables ────────────────────────────────────────────────────────────────────
# Alembic creates and changes them (backend/migrations/README.md). A database
# from before Alembic has to be marked once, or this stops: alembic stamp <revision>.
echo "[job] Bringing the database to the newest migration ..."
(cd $REPO/backend && alembic upgrade head)

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
