#!/bin/bash
#SBATCH --job-name=value_mappings_to_yaml
#SBATCH --partition=intelsr_short         # A100 node (up to 1 day); use sgpu_long for >1 day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=ag_ins_rump
#SBATCH --output=/home/s24pjoha_hpc/Text_To_SQL/unmapped_vals_yaml_gemini%j.log


# run_value_mapping_tools.sh
# ---------------------------
# Starts the PostgreSQL instance (same pattern as crawl_job.sh) and then
# runs one of the three value-mapping audit / normalisation scripts.
#
# Usage:
#   bash training_data_generation/run_value_mapping_tools.sh audit
#       → runs check_yaml_codes_in_db.py (which YAML codes exist in DB?)
#         then find_unmapped_db_values.py  (which DB values are not in YAML?)
#
#   bash training_data_generation/run_value_mapping_tools.sh canonicalise [--dry-run]
#       → runs apply_canonical_codes.py (normalise DB codes to canonical form)
#         pass --dry-run to preview without modifying the DB
#
# All three scripts also accept --db-url if you want to override the default.

set -euo pipefail

# ── Configuration (mirrors crawl_job.sh) ─────────────────────────────────────
# Resolve the real path of this script so the repo root is correct even when
# submitted via SLURM (where BASH_SOURCE[0] may be relative to /var/spool/slurmd).
REPO="/home/s24pjoha_hpc/Text_To_SQL"
PIPELINE_DIR="/home/s24pjoha_hpc/Text_To_SQL/training_data_generation"
VENV="/home/s24pjoha_hpc/Text_To_SQL/BIDS-SQL/venv_hpc"
SCRATCH="/lustre/scratch/data/s24pjoha_hpc-llm_sql_data"
PG_DATA="$SCRATCH/postgres_data"
PG_PORT=5429
PG_LOG="$PG_DATA/logfile"
DB_URL="postgresql://user:password@localhost:${PG_PORT}/bids_sql"

MODE="${1:-}"
shift || true   # remaining args forwarded to the Python script

usage() {
    cat <<EOF
Usage:
  bash run_value_mapping_tools.sh audit
      → find unmapped DB values; writes unmapped_db_values.yaml + dataset_context.json

  bash run_value_mapping_tools.sh map [--resume] [--sections diagnosis task ...] [--batch-size N]
      → call Gemini to classify unmapped values; apply to value_mappings.yaml after each batch

  bash run_value_mapping_tools.sh map-generate [--resume] [--sections ...]
      → generate decisions only (no YAML changes); review checkpoint, then run map-apply

  bash run_value_mapping_tools.sh map-apply
      → apply a previously reviewed checkpoint to value_mappings.yaml

  bash run_value_mapping_tools.sh canonicalise [--dry-run]
      → apply canonical codes from value_mappings.yaml to the DB
EOF
}

if [[ -z "$MODE" ]]; then
    usage
    exit 1
fi

# ── Load modules ──────────────────────────────────────────────────────────────
module use /opt/software/easybuild-AMD/modules/all 2>/dev/null || true
module load Python/3.11.3-GCCcore-12.3.0       2>/dev/null || true

# ── Activate venv ─────────────────────────────────────────────────────────────
source "/home/s24pjoha_hpc/Text_To_SQL/BIDS-SQL/venv_hpc/bin/activate"

# ── Start PostgreSQL (only for modes that need it) ────────────────────────────
# map / map-generate / map-apply only read/write YAML and call Gemini — no DB needed.
_needs_db() {
    case "$MODE" in
        audit|canonicalise|canonicalize) return 0 ;;
        *) return 1 ;;
    esac
}

if _needs_db; then
    module load PostgreSQL/16.1-GCCcore-12.3.0 2>/dev/null || true

    if [ ! -d "$PG_DATA/base" ]; then
        echo "[pg] Cluster not found — this script does not initialise a new cluster."
        echo "     Run a crawl job first to create $PG_DATA."
        exit 1
    fi

    echo "[pg] Ensuring PostgreSQL is running on port $PG_PORT ..."
    pkill -f "postgres.*${PG_DATA}" 2>/dev/null || true
    sleep 3
    rm -f "$PG_DATA/postmaster.pid"
    pg_ctl -D "$PG_DATA" -l "$PG_LOG" -o "-p $PG_PORT" start -w -t 120

    echo "[pg] Verifying connectivity ..."
    for i in $(seq 1 10); do
        psql -h localhost -p "$PG_PORT" -d postgres -c "SELECT 1;" >/dev/null 2>&1 && break
        echo "[pg] Waiting ... ($i/10)"
        sleep 3
    done
    psql -h localhost -p "$PG_PORT" -d postgres -c "SELECT 1;" >/dev/null 2>&1 \
        || { echo "[pg] ERROR: PostgreSQL not reachable — check $PG_LOG"; exit 1; }
    echo "[pg] PostgreSQL is ready."
fi

# ── Run the requested tool ────────────────────────────────────────────────────
cd "$REPO"

case "$MODE" in
    audit)
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  STEP 1: Check which YAML codes are in the DB"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/check_yaml_codes_in_db.py" \
            --db-url "$DB_URL" "$@"

        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  STEP 2: Find DB values not covered by YAML"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/find_unmapped_db_values.py" \
            --db-url "$DB_URL" \
            --out "$PIPELINE_DIR/unmapped_db_values.yaml" \
            --dataset-context "$PIPELINE_DIR/dataset_context.json" "$@"

        echo ""
        echo "Review: $PIPELINE_DIR/unmapped_db_values.yaml"
        echo "Dataset context written to: $PIPELINE_DIR/dataset_context.json"
        echo "Next: run 'map' to integrate unmapped values via Gemini."
        ;;

    map)
        # Generate Gemini decisions AND continuously apply them to value_mappings.yaml.
        # Each batch is applied immediately so later batches see the updated tree.
        # Pass --resume to continue an interrupted run.
        # Pass --sections diagnosis task ... to limit scope.
        # Pass --batch-size N to control values per LLM call.
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Mapping unmapped DB values via Gemini (generate+apply)"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/auto_map_unmapped.py" \
            --unmapped  "$PIPELINE_DIR/unmapped_db_values.yaml" \
            --mappings  "$PIPELINE_DIR/value_mappings.yaml" \
            --checkpoint "$PIPELINE_DIR/mapping_decisions.json" \
            --log        "$PIPELINE_DIR/mapping_integration_log.jsonl" \
            --dataset-context "$PIPELINE_DIR/dataset_context.json" "$@"

        echo ""
        echo "value_mappings.yaml updated."
        echo "Integration log: $PIPELINE_DIR/mapping_integration_log.jsonl"
        echo "Checkpoint: $PIPELINE_DIR/mapping_decisions.json"
        echo "Next: run 'audit' to verify coverage, then 'canonicalise' to normalise the DB."
        ;;

    map-generate)
        # Generate decisions only — do NOT apply to value_mappings.yaml.
        # Review mapping_decisions.json, then run 'map-apply'.
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Generating mapping decisions (no YAML changes yet)"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/auto_map_unmapped.py" \
            --unmapped  "$PIPELINE_DIR/unmapped_db_values.yaml" \
            --mappings  "$PIPELINE_DIR/value_mappings.yaml" \
            --checkpoint "$PIPELINE_DIR/mapping_decisions.json" \
            --log        "$PIPELINE_DIR/mapping_integration_log.jsonl" \
            --dataset-context "$PIPELINE_DIR/dataset_context.json" \
            --generate-only "$@"

        echo ""
        echo "Decisions written to: $PIPELINE_DIR/mapping_decisions.json"
        echo "Review the checkpoint, then run 'map-apply' to update value_mappings.yaml."
        ;;

    map-apply)
        # Apply a previously generated (and reviewed) checkpoint to value_mappings.yaml.
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Applying mapping decisions to value_mappings.yaml"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/auto_map_unmapped.py" \
            --mappings  "$PIPELINE_DIR/value_mappings.yaml" \
            --checkpoint "$PIPELINE_DIR/mapping_decisions.json" \
            --log        "$PIPELINE_DIR/mapping_integration_log.jsonl" \
            --apply "$@"

        echo ""
        echo "value_mappings.yaml updated."
        echo "Integration log: $PIPELINE_DIR/mapping_integration_log.jsonl"
        ;;

    map)
        # Generate Gemini decisions AND continuously apply them to value_mappings.yaml.
        # Each batch is applied immediately so later batches see the updated tree.
        # Pass --resume to continue an interrupted run.
        # Pass --sections diagnosis task ... to limit scope.
        # Pass --batch-size N to control values per LLM call.
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Mapping unmapped DB values via Gemini (generate+apply)"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/auto_map_unmapped.py" \
            --unmapped   "$PIPELINE_DIR/unmapped_db_values.yaml" \
            --mappings   "$PIPELINE_DIR/value_mappings.yaml" \
            --checkpoint "$PIPELINE_DIR/mapping_decisions.json" \
            --log        "$PIPELINE_DIR/mapping_integration_log.jsonl" \
            --dataset-context "$PIPELINE_DIR/dataset_context.json" "$@"

        echo ""
        echo "value_mappings.yaml updated."
        echo "Integration log: $PIPELINE_DIR/mapping_integration_log.jsonl"
        echo "Checkpoint:      $PIPELINE_DIR/mapping_decisions.json"
        echo "Next: run 'audit' to verify coverage, then 'canonicalise' to normalise the DB."
        ;;

    map-generate)
        # Generate decisions only — do NOT apply to value_mappings.yaml.
        # Review mapping_decisions.json manually, then run 'map-apply'.
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Generating mapping decisions (no YAML changes yet)"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/auto_map_unmapped.py" \
            --unmapped   "$PIPELINE_DIR/unmapped_db_values.yaml" \
            --mappings   "$PIPELINE_DIR/value_mappings.yaml" \
            --checkpoint "$PIPELINE_DIR/mapping_decisions.json" \
            --log        "$PIPELINE_DIR/mapping_integration_log.jsonl" \
            --dataset-context "$PIPELINE_DIR/dataset_context.json" \
            --generate-only "$@"

        echo ""
        echo "Decisions written to: $PIPELINE_DIR/mapping_decisions.json"
        echo "Review the checkpoint, then run 'map-apply' to update value_mappings.yaml."
        ;;

    map-apply)
        # Apply a previously generated (and reviewed) checkpoint to value_mappings.yaml.
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Applying mapping decisions to value_mappings.yaml"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/auto_map_unmapped.py" \
            --mappings   "$PIPELINE_DIR/value_mappings.yaml" \
            --checkpoint "$PIPELINE_DIR/mapping_decisions.json" \
            --log        "$PIPELINE_DIR/mapping_integration_log.jsonl" \
            --apply "$@"

        echo ""
        echo "value_mappings.yaml updated."
        echo "Integration log: $PIPELINE_DIR/mapping_integration_log.jsonl"
        ;;

    canonicalise|canonicalize)
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Applying canonical codes to DB"
        echo "════════════════════════════════════════════════════"
        python "$PIPELINE_DIR/apply_canonical_codes.py" \
            --db-url "$DB_URL" "$@"
        ;;

    *)
        echo "Unknown mode: $MODE"
        usage
        exit 1
        ;;
esac

echo ""
echo "[done]"
