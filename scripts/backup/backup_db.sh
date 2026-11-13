#!/usr/bin/env bash
# Nightly off-site backup of the BIDS-Eye database (and the deployment's .env)
# to a Hetzner Storage Box, using BorgBackup.
#
# Borg stores everything encrypted, compressed and deduplicated: consecutive
# dumps share most of their data, so dozens of daily copies of a 3.7 GB
# database fit in a few GB. Nothing is written to the server's own disk.
#
# Configuration: /root/.config/bidseye-backup.env (see README.md), with
#   BORG_REPO        ssh://uNNNNN@uNNNNN.your-storagebox.de:23/./bidseye
#   BORG_PASSPHRASE  repository passphrase (keep a copy outside the server!)
#   BORG_RSH         ssh -i /root/.ssh/id_storagebox -o BatchMode=yes
# Optional: PG_CONTAINER (default bids_postgres), PROJECT_DIR (default /home/BIDS-Eye)
#
# Run by bidseye-backup.timer; can also be run by hand.
set -euo pipefail

ENV_FILE="${BIDSEYE_BACKUP_ENV:-/root/.config/bidseye-backup.env}"
# shellcheck disable=SC1090
source "$ENV_FILE"
: "${BORG_REPO:?BORG_REPO missing in $ENV_FILE}"
: "${BORG_PASSPHRASE:?BORG_PASSPHRASE missing in $ENV_FILE}"
export BORG_REPO BORG_PASSPHRASE BORG_RSH

PG_CONTAINER="${PG_CONTAINER:-bids_postgres}"
PROJECT_DIR="${PROJECT_DIR:-/home/BIDS-Eye}"
STATUS_DIR=/var/lib/bidseye-backup
mkdir -p "$STATUS_DIR"

log() { echo "$(date -Is) $*"; }

# 1) Database. The dump is written uncompressed (-Z 0) on purpose: borg
#    compresses itself, and a compressed dump would defeat deduplication.
#    --content-from-command makes borg check pg_dump's exit status: a failed
#    or interrupted dump never becomes an archive.
log "dumping database from $PG_CONTAINER"
borg create --stats --compression zstd,6 \
    --stdin-name bids_sql.dump \
    --content-from-command \
    "::bids_sql-{now:%Y-%m-%dT%H%M}" \
    -- docker exec "$PG_CONTAINER" sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -Z 0'

# 2) Deployment configuration needed to bring the service back: secrets in
#    .env and the compose/nginx files. Small, so a copy every night is cheap.
log "saving deployment configuration"
borg create --stats --compression zstd,6 \
    "::config-{now:%Y-%m-%dT%H%M}" \
    "$PROJECT_DIR/.env" \
    "$PROJECT_DIR/docker-compose-hetzner.yml" \
    "$PROJECT_DIR/nginx"

# 3) Retention: 14 daily, 8 weekly, 24 monthly copies of each kind.
log "pruning old archives"
for prefix in bids_sql config; do
    borg prune --list --glob-archives "${prefix}-*" \
        --keep-daily 14 --keep-weekly 8 --keep-monthly 24
done
borg compact

date -Is > "$STATUS_DIR/last_success"
log "backup finished"
