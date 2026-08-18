#!/usr/bin/env python3
"""Write a compact, human-readable progress line for every completed step."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-.*?\btiming_s/step:(?:np\.float64\()?([0-9.eE+-]+)")
TOTAL_STEPS_RE = re.compile(r"['\"]?total_training_steps['\"]?\s*[:=]\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--initial-step", type=int, default=0)
    parser.add_argument("--start-epoch", required=True, type=float)
    return parser.parse_args()


def duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_bar(step: int, total: int, width: int = 30) -> str:
    completed = min(width, round(width * step / total))
    return "█" * completed + "░" * (width - completed)


def format_line(
    step: int, total: int, step_seconds: float, start_epoch: float, initial_step: int
) -> str:
    now = time.time()
    elapsed = max(0.0, now - start_epoch)
    completed_this_run = step - initial_step
    remaining = elapsed / completed_this_run * (total - step) if completed_this_run > 0 else 0.0
    eta = dt.datetime.fromtimestamp(now + remaining, tz=dt.timezone.utc)
    timestamp = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc)
    percentage = 100.0 * step / total
    return (
        f"{timestamp:%Y-%m-%d %H:%M:%S} UTC | "
        f"[{progress_bar(step, total)}] {percentage:6.2f}% | "
        f"Step {step:>4}/{total:<4} | "
        f"累计 {duration(elapsed):>11} | 本步 {duration(step_seconds):>8} | "
        f"剩余 ≈ {duration(remaining):>11} | ETA {eta:%Y-%m-%d %H:%M:%S} UTC"
    )


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_step = args.initial_step
    output_mode = "a" if args.initial_step > 0 else "w"
    with args.output.open(output_mode, encoding="utf-8", buffering=1) as output:
        total_steps = None
        for raw_line in sys.stdin:
            line = ANSI_RE.sub("", raw_line)
            if total_steps is None:
                total_match = TOTAL_STEPS_RE.search(line)
                if total_match:
                    total_steps = int(total_match.group(1))
                    output.write(
                        f"训练进度 | Run: {args.run_name} | Total steps: {total_steps} | "
                        f"Initial step: {args.initial_step}\n"
                    )
                    output.write("=" * 150 + "\n")
            match = STEP_RE.search(line)
            if not match or total_steps is None:
                continue
            step = int(match.group(1))
            if step <= last_step or step > total_steps:
                continue
            output.write(
                format_line(step, total_steps, float(match.group(2)), args.start_epoch, args.initial_step) + "\n"
            )
            last_step = step
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
