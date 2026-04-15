#!/usr/bin/env bash
# scripts/upload_weights.sh
# -------------------------
# Upload local LoRA adapter weights to the Modal Volume used by the
# BIDS-Eye inference function.
#
# Usage:
#   bash scripts/upload_weights.sh                   # uses ./model_weights/
#   bash scripts/upload_weights.sh /path/to/weights  # custom path
#
# Run once before deploying:
#   bash scripts/upload_weights.sh
#   modal deploy modal_app/app.py

set -euo pipefail

VOLUME_NAME="bids-eye-weights"
LOCAL_DIR="${1:-model_weights}"
REMOTE_DIR="/adapters"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "ERROR: directory '$LOCAL_DIR' not found."
    echo "Put your QLoRA adapter files there first, then re-run."
    echo "  Expected files: adapter_config.json, adapter_model.safetensors (or .bin)"
    exit 1
fi

echo "==> Creating Modal volume '$VOLUME_NAME' (no-op if it already exists)..."
modal volume create "$VOLUME_NAME" 2>/dev/null || true

echo "==> Uploading '$LOCAL_DIR/' → modal://$VOLUME_NAME$REMOTE_DIR ..."
modal volume put "$VOLUME_NAME" "$LOCAL_DIR/" "$REMOTE_DIR"

echo ""
echo "Done. Next step:"
echo "  modal deploy modal_app/app.py"
