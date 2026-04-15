#!/bin/bash
#SBATCH --job-name=create-train-venv
#SBATCH --partition=intelsr_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs/create_venv_%j.log

set -euo pipefail

SCRATCH=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data
VENV=$SCRATCH/train_venv
TMP_VENV=$SCRATCH/train_venv.tmp.${SLURM_JOB_ID:-$$}
OLD_VENV=$SCRATCH/train_venv.old.${SLURM_JOB_ID:-$$}

module use /opt/software/easybuild-AMD/modules/all
module load Python/3.11.3-GCCcore-12.3.0

cleanup() {
    set +e
    if [ -d "$TMP_VENV" ]; then
        rm -rf "$TMP_VENV"
    fi
    if [ -d "$OLD_VENV" ]; then
        if [ -d "$VENV" ]; then
            rm -rf "$OLD_VENV"
        else
            mv "$OLD_VENV" "$VENV"
        fi
    fi
}

trap cleanup EXIT

mkdir -p "$SCRATCH/logs"

if [ -d "$TMP_VENV" ]; then
    echo "Removing stale temporary venv..."
    rm -rf "$TMP_VENV"
fi

echo "Creating temporary venv at $TMP_VENV ..."
python -m venv "$TMP_VENV"
source "$TMP_VENV/bin/activate"
python -m pip install --upgrade pip --quiet

echo "Installing PyTorch (CUDA 12.1)..."
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing training libs..."
python -m pip install --quiet \
    transformers==4.44.0 \
    peft==0.12.0 \
    trl==0.10.1 \
    bitsandbytes==0.43.3 \
    accelerate==0.34.0 \
    datasets \
    sentence-transformers \
    sqlparse \
    sqlalchemy \
    psycopg2-binary \
    rich

echo "Verifying required imports..."
python - <<'PY'
import importlib.util
from importlib.metadata import version

required = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "sentence_transformers": "sentence-transformers",
    "sqlparse": "sqlparse",
    "sqlalchemy": "sqlalchemy",
    "psycopg2": "psycopg2-binary",
    "rich": "rich",
}

missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    raise SystemExit(
        "[env] ERROR: venv is missing required packages after install: "
        + ", ".join(missing)
    )

for mod, pkg in required.items():
    print(f"{pkg:<24} {version(pkg)}")
PY

if [ -d "$VENV" ]; then
    echo "Swapping verified venv into place..."
    mv "$VENV" "$OLD_VENV"
fi

mv "$TMP_VENV" "$VENV"
rm -rf "$OLD_VENV"

echo "Done. Venv ready at $VENV"
