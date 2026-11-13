# Off-site database backup

Nightly, encrypted, deduplicated backups of the production database (and the
deployment's `.env`, compose file and nginx config) to a Hetzner Storage Box,
using [BorgBackup](https://borgbackup.readthedocs.io/). Nothing is stored on
the server's own disk.

| | |
|---|---|
| What | `pg_dump` of `bids_sql` (3.7 GB on disk, ~650 MB compressed), plus deployment config |
| Where | Hetzner Storage Box BX11 (1 TB, about €3.20/month excl. VAT), in a different data centre than the server |
| When | daily 03:30 server time (`bidseye-backup.timer`) |
| Kept | 14 daily, 8 weekly, 24 monthly copies |
| Checked | monthly restore into a throwaway container (`bidseye-restore-test.timer`) |

Tested locally on 2026-11-13 with borg 1.4.4 (the version Ubuntu 26.04
ships): backup, a second deduplicated backup, a failing dump that correctly
leaves no archive, and a restore test.

## One-time setup

1. **Order a Storage Box** (Hetzner Console → Storage Boxes, BX11) in FSN1 or
   HEL1. In its settings enable *SSH support* and *External reachability*.
   Note the user name (`uNNNNN`) and host (`uNNNNN.your-storagebox.de`).

2. **On the server**, install borg and create an SSH key for the box:

   ```bash
   apt-get install -y borgbackup
   ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_storagebox
   # upload the public key (asks for the Storage Box password once)
   cat /root/.ssh/id_storagebox.pub | ssh -p 23 uNNNNN@uNNNNN.your-storagebox.de install-ssh-key
   ```

3. **Write the config** `/root/.config/bidseye-backup.env` (mode 600):

   ```bash
   BORG_REPO=ssh://uNNNNN@uNNNNN.your-storagebox.de:23/./bidseye
   BORG_PASSPHRASE=<long random passphrase>
   BORG_RSH="ssh -i /root/.ssh/id_storagebox -o BatchMode=yes"
   ```

4. **Create the repository** and export its key:

   ```bash
   set -a; . /root/.config/bidseye-backup.env; set +a
   borg init --encryption=repokey-blake2
   borg key export --paper > /root/bidseye-borg-key.txt
   ```

   **Store the passphrase and the exported key outside the server** (password
   manager). Without both, the backups cannot be decrypted if the server is lost.

5. **Install the timers** and run the first backup by hand:

   ```bash
   cp /home/BIDS-Eye/scripts/backup/*.service /home/BIDS-Eye/scripts/backup/*.timer /etc/systemd/system/
   systemctl daemon-reload
   systemctl enable --now bidseye-backup.timer bidseye-restore-test.timer
   systemctl start bidseye-backup.service && journalctl -u bidseye-backup -n 30
   systemctl start bidseye-restore-test.service && journalctl -u bidseye-restore-test -n 10
   ```

## Day to day

```bash
systemctl list-timers 'bidseye-*'               # next runs
cat /var/lib/bidseye-backup/last_success        # last good backup
journalctl -u bidseye-backup --since yesterday  # log of the last run
set -a; . /root/.config/bidseye-backup.env; set +a; borg list   # all archives
```

## Restoring production

```bash
set -a; . /root/.config/bidseye-backup.env; set +a
borg list --glob-archives 'bids_sql-*'                       # pick an archive
docker compose -f docker-compose-hetzner.yml stop backend    # no writes during restore
borg extract --stdout ::bids_sql-YYYY-MM-DDTHHMM bids_sql.dump \
  | docker exec -i bids_postgres sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner'
docker compose -f docker-compose-hetzner.yml start backend
```

`borg extract ::config-YYYY-MM-DDTHHMM` restores `.env`, the compose file and
the nginx config into the current directory.

## Optional extra: Hetzner Cloud Backups

The Cloud Console can also take whole-server images (7 slots, 20% of the
server price). They help to roll the whole machine back after a bad update,
but keep only a week and live in the same provider account and region, so
they complement the Storage Box rather than replace it.
