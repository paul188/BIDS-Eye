#!/bin/bash
#SBATCH --job-name=bids-sql-train
#SBATCH --partition=sgpu_medium         # A100 node (up to 1 day); use sgpu_long for >1 day
#SBATCH --gres=gpu:a100:1               # 1× A100 (80 GB) — plenty for 7B QLoRA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --account=ag_ins_rump
#SBATCH --output=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs/train_%j.log
#SBATCH --error=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs/train_%j.log

set -e

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO=/home/s24pjoha_hpc/Text_To_SQL
SCRATCH=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data
TRAIN_SCRIPT=$REPO/training/train.py
DATA=$SCRATCH/training.jsonl
MODEL_SRC=$SCRATCH/model_weights/sqlcoder-7b-2   # downloaded by download_weights.sh
OUTPUT=$SCRATCH/checkpoints/run_$(date +%Y%m%d_%H%M)
VENV=$SCRATCH/train_venv

# ── HuggingFace cache → Lustre (never fills home quota) ───────────────────────
export HF_HOME=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
mkdir -p $HF_HOME $HF_DATASETS_CACHE $OUTPUT \
         /lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs

# ── Modules ────────────────────────────────────────────────────────────────────
module use /opt/software/easybuild-AMD/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load PostgreSQL/16.1-GCCcore-12.3.0

# ── Virtual environment ────────────────────────────────────────────────────────
# Venv must be created on a CPU node first (GPU nodes have incompatible CPU arch).
# Run create_train_venv.sh on intelsr_short before submitting this job.
if [ ! -d "$VENV" ]; then
    echo "[train] ERROR: venv not found at $VENV"
    echo "[train] Run: sbatch create_train_venv.sh"
    exit 1
fi
source $VENV/bin/activate

echo "[train] Python:  $(python --version)"
echo "[train] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[train] GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "[train] VRAM:    $(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')")"

# ── Copy model to local node SSD if available ──────────────────────────────────
# Lustre has high latency for many small reads (model shards). Copying to the
# node-local SSD ($TMPDIR) before loading can cut startup time significantly.
MODEL_PATH=$MODEL_SRC
if [ -d "$TMPDIR" ] && [ "$(df -BG $TMPDIR | awk 'NR==2{print $4}' | tr -d G)" -gt 20 ]; then
    echo "[train] Copying model to local SSD ($TMPDIR) ..."
    cp -r $MODEL_SRC $TMPDIR/model
    MODEL_PATH=$TMPDIR/model
    echo "[train] Done copying."
fi

# ── PostgreSQL for execution-match eval ────────────────────────────────────────
# The crawl job populates $SCRATCH/postgres_data on Lustre (shared filesystem).
# We always snapshot the data dir to node-local $TMPDIR so we can start our own
# pg instance regardless of whether the crawl job is currently running.
# (Two pg_ctl processes cannot share one data directory — the snapshot avoids this.)
PG_DATA=$SCRATCH/postgres_data
PG_SNAP=$TMPDIR/pg_train_snap   # node-local SSD — fast & isolated
PG_PORT=5430
PG_LOG=$SCRATCH/logs/pg_train_$SLURM_JOB_ID.log
DB_URL=""

if [ ! -d "$PG_DATA/base" ]; then
    echo "[train] No DB at $PG_DATA — execution-match eval disabled (run a crawl job first)."
else
    echo "[train] Snapshotting DB to local SSD ($PG_SNAP) ..."
    rsync -a --exclude='postmaster.pid' --exclude='postmaster.opts' \
          $PG_DATA/ $PG_SNAP/
    echo "[train] Snapshot done. Starting PostgreSQL on port $PG_PORT ..."
    pg_ctl -D $PG_SNAP -l $PG_LOG -o "-p $PG_PORT" start -w -t 60
    if psql -h localhost -p $PG_PORT -U user -d bids_sql \
            -c "SELECT COUNT(*) FROM bids_datasets;" > /dev/null 2>&1; then
        DATASET_COUNT=$(psql -h localhost -p $PG_PORT -U user -d bids_sql -tAc "SELECT COUNT(*) FROM bids_datasets;")
        echo "[train] DB ready — ${DATASET_COUNT} datasets — execution-match eval enabled."
        DB_URL="postgresql://user:password@localhost:$PG_PORT/bids_sql"
    else
        echo "[train] WARN: DB failed to start. Check $PG_LOG"
        pg_ctl -D $PG_SNAP stop -m fast 2>/dev/null || true
    fi
fi

# ── Train ──────────────────────────────────────────────────────────────────────
echo "[train] Starting at $(date)"
echo "[train] Data:   $DATA"
echo "[train] Model:  $MODEL_PATH"
echo "[train] Output: $OUTPUT"

DB_ARG=""
[ -n "$DB_URL" ] && DB_ARG="--db-url $DB_URL"

python $TRAIN_SCRIPT \
    --model   "$MODEL_PATH" \
    --data    "$DATA" \
    --output  "$OUTPUT" \
    --epochs  3 \
    --batch   2 \
    --grad-accum 16 \
    --lr      2e-4 \
    --lora-r  32 \
    --max-len 1024 \
    --load-in-4bit \
    $DB_ARG

# ── Stop DB ────────────────────────────────────────────────────────────────────
[ -n "$DB_URL" ] && pg_ctl -D $PG_SNAP stop -m fast 2>/dev/null || true

echo "[train] Finished at $(date)"
echo "[train] Checkpoints at $OUTPUT"
