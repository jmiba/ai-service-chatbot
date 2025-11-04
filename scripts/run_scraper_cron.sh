#!/bin/bash
set -euo pipefail

INTERVAL_SECONDS="${SCRAPER_INTERVAL_SECONDS:-43200}"

log() {
  printf '[scraper-cron] %s\n' "$1"
}

while true; do
  log "Starting scheduled scrape (interval ${INTERVAL_SECONDS}s)…"
  if python scripts/cli_scrape.py --mode both; then
    log "Scrape run finished successfully."
  else
    log "Scrape run failed – will retry after sleep."
  fi
  log "Sleeping for ${INTERVAL_SECONDS}s before next run."
  sleep "${INTERVAL_SECONDS}"
done
