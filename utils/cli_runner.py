"""Helpers for launching the CLI scraper/vectorizer out-of-process."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional

try:  # Streamlit is only available when running inside the app
    from streamlit.runtime.scriptrunner import (
        RerunData,
        add_script_run_ctx,
        get_script_run_ctx,
    )
except Exception:  # pragma: no cover - streamlit not installed in some environments
    RerunData = None  # type: ignore[assignment]
    add_script_run_ctx = None  # type: ignore[assignment]
    get_script_run_ctx = None  # type: ignore[assignment]


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
    auto_rerun: bool = False,
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
    auto_rerun:
        When True and running inside Streamlit, enqueue a rerun when new log output
        arrives so the UI updates without manual refreshes.
    """

    script_path = _ensure_cli_script()

    if mode not in {"scrape", "vectorize", "both"}:
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

    rerun_ctx = None
    if auto_rerun and RerunData and get_script_run_ctx:
        try:
            rerun_ctx = get_script_run_ctx()  # type: ignore[call-arg]
        except Exception:
            rerun_ctx = None
    rerun_enabled = rerun_ctx is not None

    def _reader() -> None:
        last_rerun_ts = 0.0
        rerun_interval = 0.4  # throttle reruns so we do not spin the UI

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
                if rerun_enabled and getattr(rerun_ctx, "script_requests", None):
                    now = time.monotonic()
                    if now - last_rerun_ts > rerun_interval:
                        try:
                            rerun_ctx.script_requests.request_rerun(  # type: ignore[attr-defined]
                                RerunData(is_auto_rerun=True)  # type: ignore[call-arg]
                            )
                            last_rerun_ts = now
                        except Exception:
                            pass

    thread = threading.Thread(target=_reader, name=f"cli-scrape-{mode}", daemon=True)
    if rerun_enabled and add_script_run_ctx:
        try:
            add_script_run_ctx(thread, rerun_ctx)  # type: ignore[arg-type]
        except Exception:
            pass
    thread.start()

    return CLIJob(process=process, mode=mode, start_time=datetime.utcnow(), logs=logs, _reader=thread)
