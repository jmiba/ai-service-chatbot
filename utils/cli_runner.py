"""Helpers for launching the CLI scraper/vectorizer out-of-process."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
CLI_SCRIPT = BASE_DIR / "scripts" / "cli_scrape.py"


class CLIJobError(RuntimeError):
    """Raised when a CLI job cannot be launched."""


@dataclass
class CLIJob:
    """Represents a running or finished cli_scrape.py process."""

    process: subprocess.Popen[str]
    mode: str
    start_time: datetime
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=500))
    _reader: Optional[threading.Thread] = field(default=None, repr=False)

    def is_running(self) -> bool:
        return self.process.poll() is None

    def returncode(self) -> Optional[int]:
        return self.process.poll()

    def terminate(self) -> None:
        if self.is_running():
            self.process.terminate()


def _ensure_cli_script() -> Path:
    if not CLI_SCRIPT.exists():
        raise CLIJobError(f"cli_scrape.py not found at {CLI_SCRIPT}")
    return CLI_SCRIPT


def launch_cli_job(
    *,
    mode: str,
    args: Optional[Iterable[str]] = None,
    env_overrides: Optional[dict[str, str]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    max_log_lines: int = 500,
) -> CLIJob:
    """Spawn cli_scrape.py with the provided mode in a background process.

    Parameters
    ----------
    mode:
        One of the values accepted by cli_scrape.py's --mode flag.
    args:
        Additional command-line arguments to pass to the CLI (e.g. crawl budget).
    env_overrides:
        Optional environment variables to merge with the current process environment.
    log_callback:
        Called for each stdout line emitted by the CLI.
    max_log_lines:
        Retained line count for the job's in-memory log buffer.
    """

    script_path = _ensure_cli_script()

    if mode not in {"scrape", "vectorize", "all", "cleanup"}:
        raise CLIJobError(f"Unsupported mode '{mode}'.")

    python_exe = sys.executable
    if not python_exe:
        raise CLIJobError("Cannot determine Python executable for launching CLI job.")

    cmd = [python_exe, str(script_path), "--mode", mode]
    if args:
        cmd.extend(args)

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR),
            env=env,
        )
    except OSError as exc:  # pragma: no cover - depends on runtime env
        raise CLIJobError(f"Failed to start cli_scrape.py: {exc}") from exc

    logs: Deque[str] = deque(maxlen=max_log_lines)

    def _reader() -> None:
        assert process.stdout is not None
        with process.stdout:
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                logs.append(line)
                if log_callback:
                    try:
                        log_callback(line)
                    except Exception:
                        pass

    thread = threading.Thread(target=_reader, name=f"cli-scrape-{mode}", daemon=True)
    thread.start()

    return CLIJob(process=process, mode=mode, start_time=datetime.utcnow(), logs=logs, _reader=thread)
