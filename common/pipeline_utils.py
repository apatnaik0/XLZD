"""
Functions for pipeline utility and timing
"""
from __future__ import annotations

import time


def log_stage(message: str) -> float:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    return time.perf_counter()


def finish_stage(stage_start: float, message: str) -> None:
    elapsed = time.perf_counter() - stage_start
    print(f"[done in {elapsed:.2f}s] {message}", flush=True)