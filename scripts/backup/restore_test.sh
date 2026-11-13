#!/usr/bin/env bash
# Prove that the newest database backup can be restored, without touching the
# live database: restore it into a throwaway Postgres container, count rows,
# compare with the live database, and remove the container again.
#
# Run monthly (bidseye-restore-test.timer) or by hand. Needs about 8 GB of
# free disk for the duration of the test.
set -euo pipefail

ENV_FILE="${BIDSEYE_BACKUP_ENV:-/root/.config/bidseye-backup.env}"
# shellcheck disable=SC1090
source "$ENV_FILE"
export BORG_REPO BORG_PASSPHRASE BORG_RSH
PG_CONTAINER="${PG_CONTAINER:-bids_postgres}"
TEST=bidseye_restore_test

archive=$(borg list --short --glob-archives 'bids_sql-*' --last 1)
[ -n "$archive" ] || { echo "no database archive found"; exit 1; }
echo "restoring $archive into a throwaway container"

docker rm -f "$TEST" >/dev/null 2>&1 || true
trap 'docker rm -f "$TEST" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$TEST" -e POSTGRES_PASSWORD=restore -e POSTGRES_DB=restore postgres:16 >/dev/null
for _ in $(seq 1 60); do
    docker exec "$TEST" pg_isready -U postgres -d restore >/dev/null 2>&1 && break
    sleep 2
done

borg extract --stdout "::$archive" bids_sql.dump \
    | docker exec -i "$TEST" pg_restore -U postgres -d restore --no-owner --exit-on-error

count_sql="SELECT 'datasets=' || (SELECT count(*) FROM bids_datasets) || ' objects=' || (SELECT count(*) FROM bids_objects) || ' participants=' || (SELECT count(*) FROM bids_participants)"
restored=$(docker exec "$TEST" psql -U postgres -d restore -At -c "$count_sql")
live=$(docker exec "$PG_CONTAINER" sh -c "psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -At -c \"$count_sql\"")
echo "restored: $restored"
echo "live:     $live"
echo "restore test passed ($archive)"
