#!/usr/bin/env python3
"""Run cli_scrape.py only when the scheduled interval has elapsed."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.scraper_schedule import read_scraper_schedule, update_last_scrape_run  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cli_scrape.py if the schedule is due.")
    parser.add_argument("--mode", choices=["scrape", "vectorize", "sync", "all", "cleanup"], help="Override job mode.")
    parser.add_argument("--budget", type=int, help="Override crawl budget for scrape runs.")
    parser.add_argument("--keep-query", type=str, help="Override keep-query keys (comma separated).")
    parser.add_argument(
        "--dry-run",
        action="store_const",
        const=True,
        dest="dry_run",
        default=None,
        help="Force dry-run mode.",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_const",
        const=False,
        dest="dry_run",
        help="Disable dry-run (scheduled default may enable it).",
    )
    parser.add_argument("--force", action="store_true", help="Ignore timing and run immediately (unless disabled).")
    parser.add_argument(
        "--ignore-disabled",
        action="store_true",
        help="Run even if the schedule is currently disabled (useful for manual overrides).",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        dest="extra_args",
        default=[],
        help="Extra arguments passed verbatim to cli_scrape.py (repeatable).",
    )
    return parser.parse_args()


def _iso_to_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        dt_obj = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return dt_obj


def _format_timedelta(delta: dt.timedelta) -> str:
    total_minutes = int(delta.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _coerce_timezone(name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo((name or "").strip() or "UTC")
    except Exception:
        return ZoneInfo("UTC")


def _convert_time_strings(
    values: list[str],
    source_tz: dt.tzinfo,
    target_tz: dt.tzinfo,
) -> list[str]:
    if not values:
        return []
    today = dt.datetime.now(target_tz).date()
    converted: list[str] = []
    seen: set[str] = set()
    for token in values:
        if not token:
            continue
        try:
            hour, minute = (int(part) for part in token.split(":", 1))
        except ValueError:
            continue
        source_dt = dt.datetime.combine(today, dt.time(hour=hour, minute=minute, tzinfo=source_tz))
        target_dt = source_dt.astimezone(target_tz)
        value = target_dt.strftime("%H:%M")
        if value not in seen:
            seen.add(value)
            converted.append(value)
    return sorted(converted)


def _parse_run_times(values: list[str] | None, tz: ZoneInfo | None = None) -> list[dt.time]:
    result: list[dt.time] = []
    if not values:
        return result
    tz = tz or ZoneInfo("UTC")
    for item in values:
        if not item:
            continue
        try:
            hour, minute = (int(part) for part in item.split(":", 1))
        except ValueError:
            continue
        if 0 <= hour < 24 and 0 <= minute < 60:
            result.append(dt.time(hour=hour, minute=minute, tzinfo=tz))
    return sorted(result)


def _next_slot_after(
    last_run: dt.datetime | None,
    run_times: list[dt.time],
    now: dt.datetime,
    tz: ZoneInfo,
) -> dt.datetime | None:
    """Return the next scheduled datetime strictly after `reference`."""

    if not run_times:
        return None

    reference = last_run or now
    reference_local = reference.astimezone(tz)
    start_date = (reference_local - dt.timedelta(days=1)).date()
    end_date = reference_local.date() + dt.timedelta(days=3)

    candidates: list[dt.datetime] = []
    current = start_date
    while current <= end_date:
        for run_time in run_times:
            candidates.append(dt.datetime.combine(current, run_time))
        current += dt.timedelta(days=1)

    candidates.sort()
    for slot in candidates:
        if slot > reference:
            return slot

    future_date = end_date + dt.timedelta(days=1)
    return dt.datetime.combine(future_date, run_times[0])


def main() -> int:
    args = _parse_args()

    schedule = read_scraper_schedule()
    if not schedule["enabled"] and not args.ignore_disabled:
        print("[schedule] Runner disabled — exiting.")
        return 0

    now = dt.datetime.now(dt.timezone.utc)
    last_run = _iso_to_dt(schedule.get("last_run_at"))
    run_tz = _coerce_timezone(schedule.get("timezone"))
    raw_run_times = schedule.get("run_times")
    if schedule.get("run_times_are_utc", False):
        run_times_local = _convert_time_strings(raw_run_times or [], dt.timezone.utc, run_tz)
    else:
        run_times_local = raw_run_times
    run_times = _parse_run_times(run_times_local, tz=run_tz)

    next_slot = None
    if run_times:
        next_slot = _next_slot_after(last_run, run_times, now, run_tz)
        due = bool(next_slot and now >= next_slot)
    else:
        interval_hours = float(schedule.get("interval_hours", 12.0))
        interval = dt.timedelta(hours=interval_hours)
        due = last_run is None or now - last_run >= interval
        next_slot = last_run + interval if last_run else now + dt.timedelta(seconds=0)

    if not due and not args.force:
        remaining = next_slot - now if next_slot else dt.timedelta(0)
        print(
            f"[schedule] Next run due in {_format_timedelta(remaining)} (at {next_slot.isoformat(timespec='seconds')}).",
        )
        return 0

    mode = args.mode or schedule.get("mode", "all")
    crawl_budget = args.budget or schedule.get("crawl_budget")
    keep_query = args.keep_query if args.keep_query is not None else schedule.get("keep_query", "")
    dry_run = schedule.get("dry_run", False) if args.dry_run is None else args.dry_run

    cli_path = ROOT / "scripts" / "cli_scrape.py"
    cmd = [sys.executable or "python3", str(cli_path), "--mode", mode]

    if crawl_budget:
        cmd.extend(["--budget", str(crawl_budget)])
    if keep_query:
        cmd.extend(["--keep-query", keep_query])
    if dry_run:
        cmd.append("--dry-run")

    if args.extra_args:
        cmd.extend(args.extra_args)

    print(f"[schedule] Launching cli_scrape.py with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)

    if result.returncode == 0:
        update_last_scrape_run(now)
        print("[schedule] Run finished successfully.")
    else:
        print(f"[schedule] cli_scrape.py exited with {result.returncode}.")

    return result.returncode


if __name__ == "__main__":  # pragma: no cover - manual utility
    raise SystemExit(main())
