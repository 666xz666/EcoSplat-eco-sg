"""Opt-in per-stage latency profiling for the feed-forward pipeline.

Set ``PROFILE_STAGES=1`` to enable. Every call site is a no-op when the
variable is unset, so training and evaluation are unaffected by default.

Usage at a call site::

    _t = stage_prof.tick()
    ...work...
    stage_prof.tock("backbone", _t)

``tick`` / ``tock`` synchronise the device so the measurement covers the
work itself rather than its launch. The first ``PROFILE_STAGES_SKIP``
forwards are dropped: the first call of each shape pays for gsplat's JIT
compile and is not representative.

Results are accumulated in process and written to
``PROFILE_STAGES_OUT`` (default ``stage_profile.json``) at exit.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from typing import Dict

import torch

_ENABLED = os.environ.get("PROFILE_STAGES", "") not in ("", "0")
_OUT = os.environ.get("PROFILE_STAGES_OUT", "stage_profile.json")
_SKIP = int(os.environ.get("PROFILE_STAGES_SKIP", "3"))


class _StageProf:
    def __init__(self) -> None:
        self.sum: Dict[str, float] = {}
        self.count: Dict[str, int] = {}
        self.marks: Dict[str, float] = {}
        self.skip_left = _SKIP
        self.recording = False

    def add(self, name: str, dt: float) -> None:
        self.sum[name] = self.sum.get(name, 0.0) + dt
        self.count[name] = self.count.get(name, 0) + 1

    def note(self, name: str, value: float) -> None:
        self.marks[name] = float(value)

    def dump(self) -> None:
        payload = {
            "sum_s": self.sum,
            "count": self.count,
            "mean_ms": {
                k: 1e3 * v / max(self.count[k], 1) for k, v in self.sum.items()
            },
            "marks": self.marks,
            "skip_frames": _SKIP,
        }
        with open(_OUT, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"[stage-prof] wrote {_OUT}", flush=True)
        for key in sorted(payload["mean_ms"]):
            print(
                f"[stage-prof] {key:>12s} {payload['mean_ms'][key]:9.3f} ms "
                f"(n={payload['count'][key]})",
                flush=True,
            )
        for key in sorted(payload["marks"]):
            print(f"[stage-prof] note {key} = {payload['marks'][key]:.4f}", flush=True)


_PROF = _StageProf()
if _ENABLED:
    atexit.register(_PROF.dump)


def tick() -> float:
    """Start of a measured region; returns the timestamp."""
    if not _ENABLED:
        return 0.0
    torch.cuda.synchronize()
    return time.perf_counter()


def tock(name: str, start: float) -> None:
    """End of a measured region; records the elapsed time under ``name``."""
    if not _ENABLED:
        return
    torch.cuda.synchronize()
    dt = time.perf_counter() - start
    if name == "backbone":
        # The backbone runs once per forward, so it marks the frame boundary.
        _PROF.recording = _PROF.skip_left <= 0
        if _PROF.skip_left > 0:
            _PROF.skip_left -= 1
    if _PROF.recording:
        _PROF.add(name, dt)


def note(name: str, value: float) -> None:
    if _ENABLED:
        _PROF.note(name, value)
