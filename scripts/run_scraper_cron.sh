#!/bin/bash
set -euo pipefail

INTERVAL_SECONDS="${SCRAPER_INTERVAL_SECONDS:-43200}"
STATE_DIR="${SCRAPER_STATE_DIR:-/app/state}"
HEARTBEAT_FILE="${STATE_DIR}/last_scraper_ok"

mkdir -p "${STATE_DIR}"

log() {
  printf '[scraper-cron] %s\n' "$1"
}

while true; do
  log "Starting scheduled scrape (interval ${INTERVAL_SECONDS}s)…"
  if python scripts/cli_scrape.py --mode both; then
    log "Scrape run finished successfully."
    touch "${HEARTBEAT_FILE}"
  else
    log "Scrape run failed – will retry after sleep."
    rm -f "${HEARTBEAT_FILE}"
  fi
  log "Sleeping for ${INTERVAL_SECONDS}s before next run."
  sleep "${INTERVAL_SECONDS}"
done
