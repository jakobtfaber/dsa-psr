"""
adaptive.py  –  simple residual‑driven cadence refinement
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict
from population import Pulsar
from simulate import simulate_residuals


def refine(pop: List[Pulsar],
           schedule: Dict[int, List[int]],
           tint: Dict[int, List[float]],
           years: int,
           rng):
    """
    • For binaries or high‑Pdot pulsars, insert day+1 bursts
    • For any pulsar with forecast phase‑drift >0.25 rotations, bisect gaps
    Integration time of inserted visits = 600 s.
    """
    total_days = years * 365
    for p in pop:
        if p.is_binary or p.Pdot > 1e-14:
            extras = []
            for d in schedule[p.pid]:
                if d + 1 < total_days and rng.random() > 0.15:
                    extras.append(d + 1)
            schedule[p.pid].extend(extras)
            tint[p.pid].extend([600.0] * len(extras))

        # drift forecast
        days = sorted(schedule[p.pid])
        if len(days) < 2:
            continue
        gaps = np.diff(days)
        max_gap = gaps.max()
        phase_drift = (p.Pdot / p.P) * (max_gap * 86400)
        if phase_drift > 0.25:
            insert = []
            for d1, d2 in zip(days[:-1], days[1:]):
                if d2 - d1 > 0.75 * max_gap and d1 + 1 < total_days:
                    insert.append(d1 + (d2 - d1) // 2)
            schedule[p.pid].extend(insert)
            tint[p.pid].extend([600.0] * len(insert))

        schedule[p.pid].sort()
