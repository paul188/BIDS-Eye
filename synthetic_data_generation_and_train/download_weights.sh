#!/bin/bash
# download_weights.sh — Download model weights to Lustre scratch.
#
# Usage:
#   bash download_weights.sh                        # downloads Mistral-7B-Instruct
#   bash download_weights.sh microsoft/Phi-3-mini-4k-instruct
#
# Before running:
#   1. Create a HuggingFace account at huggingface.co
#   2. Accept the Mistral license at:
#      https://huggingface.co/microsoft/Phi-3-mini-128k-instruct  (no license required)
#   3. Generate a read token at huggingface.co/settings/tokens
#   4. Paste it when prompted (or set HF_TOKEN env var beforehand)
#
# The script uses a small SLURM devel job so it runs on a compute node
# (login nodes may have network restrictions and no Python module).
# Runtime is ~10-15 min for a 7B model on a fast network.

set -e

MODEL_ID=${1:-"microsoft/Phi-3-mini-128k-instruct"}
# Derive a clean local directory name from the model ID
MODEL_DIR_NAME=$(echo "$MODEL_ID" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | sed 's/mistralai-//' | sed 's/microsoft-//')
DEST=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/model_weights/$MODEL_DIR_NAME

echo "Model:       $MODEL_ID"
echo "Destination: $DEST"

if [ -d "$DEST" ] && [ "$(ls -A $DEST)" ]; then
    echo "Already exists. Delete $DEST to re-download."
    exit 0
fi

mkdir -p $DEST

# Ask for token if not already set
if [ -z "$HF_TOKEN" ]; then
    read -rsp "HuggingFace token (input hidden): " HF_TOKEN
    echo
fi
export HF_TOKEN

# Submit a short download job (CPU partition — no GPU needed for downloads,
# and GPU nodes have incompatible CPU architecture for the Python module)
VENV=/home/s24pjoha_hpc/Text_To_SQL/BIDS-SQL/venv_hpc

sbatch --account=ag_ins_rump --wait <<EOF
#!/bin/bash
#SBATCH --job-name=hf-download
#SBATCH --partition=intelsr_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs/download_%j.log

set -e
module load Python/3.11.3-GCCcore-12.3.0
source $VENV/bin/activate

export HF_HOME=/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/hf_cache
export HF_TOKEN=$HF_TOKEN
mkdir -p \$HF_HOME /lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs

# Install huggingface_hub into the existing venv if not present
pip install --quiet huggingface_hub

python - <<PYEOF
import os
from huggingface_hub import snapshot_download

print(f"Downloading $MODEL_ID ...")
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$DEST",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.bin"],          # skip old PyTorch shards, use safetensors
)
print(f"Done. Saved to $DEST")
PYEOF
EOF

echo ""
echo "Download complete: $DEST"
echo ""
echo "Update train_job.sh MODEL_SRC to:"
echo "  MODEL_SRC=$DEST"
