#!/usr/bin/env python3
"""Watch and safely resume the repeated Avg@16 evaluation.

The evaluator writes every completed batch atomically and is resume-safe.  This
watcher therefore only restarts it when no evaluator process exists and the
SUCCESS marker is absent.  It never kills a live process based on utilization.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "runners/evaluate_repeated_avg16.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/repeated_avg16_thinking_5datasets"
RUN_LOG = OUTPUT_ROOT / "run.log"
WATCH_LOG = OUTPUT_ROOT / "watchdog.log"
LOCK_PATH = OUTPUT_ROOT / "watchdog.lock"
SUCCESS_PATH = OUTPUT_ROOT / "SUCCESS.json"
PROCESS_MARKER = "runners/evaluate_repeated_avg16.py"
EXPECTED_SUMMARIES = 5 * 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluator_pids() -> list[int]:
    pids: list[int] = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if PROCESS_MARKER in cmdline:
            pids.append(int(proc_dir.name))
    return sorted(pids)


def gpu_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable:{exc!r}"
    return "; ".join(line.strip() for line in result.stdout.splitlines() if line.strip())


def completed_summary_count() -> int:
    return sum(1 for _ in OUTPUT_ROOT.glob("*/seed_*/summary.json"))


def run_log_age_seconds() -> float | None:
    try:
        return max(0.0, time.time() - RUN_LOG.stat().st_mtime)
    except FileNotFoundError:
        return None


def start_evaluator() -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("ab", buffering=0) as output:
        process = subprocess.Popen(
            [sys.executable, str(EVALUATOR)],
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


def inspect_once(no_restart: bool) -> bool:
    summaries = completed_summary_count()
    pids = evaluator_pids()
    age = run_log_age_seconds()
    age_text = "missing" if age is None else f"{age:.0f}s"
    gpu = gpu_snapshot()

    if SUCCESS_PATH.is_file():
        log(
            f"COMPLETE summaries={summaries}/{EXPECTED_SUMMARIES} "
            f"pids={pids} log_age={age_text} gpu=[{gpu}]"
        )
        return True

    if pids:
        log(
            f"HEALTHY summaries={summaries}/{EXPECTED_SUMMARIES} "
            f"pids={pids} log_age={age_text} gpu=[{gpu}]"
        )
        return False

    if no_restart:
        log(
            f"MISSING_NO_RESTART summaries={summaries}/{EXPECTED_SUMMARIES} "
            f"log_age={age_text} gpu=[{gpu}]"
        )
        return False

    new_pid = start_evaluator()
    log(
        f"RESTARTED pid={new_pid} summaries={summaries}/{EXPECTED_SUMMARIES} "
        f"log_age={age_text} gpu=[{gpu}]"
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Only report status; do not resume a missing evaluator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval_seconds < 60:
        raise ValueError("--interval-seconds must be at least 60")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another repeated-Avg@16 watcher is already running.")
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        log(
            f"WATCHER_START pid={os.getpid()} interval_seconds={args.interval_seconds} "
            f"python={sys.executable}"
        )
        while True:
            complete = inspect_once(no_restart=args.no_restart)
            if args.once or complete:
                return
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
