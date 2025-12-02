"""
schedule.py  –  build observing schedules with visibility & beam limits
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from population import Pulsar

_LOC = EarthLocation(lat=38.0 * u.deg, lon=-116.0 * u.deg, height=1500 * u.m)
_BEAM_MAX = 10


def _visible(p: Pulsar, day: int) -> bool:
    """Altitude >20° at local midnight of that UTC day."""
    sc = SkyCoord(l=p.l * u.deg, b=p.b * u.deg, frame='galactic').icrs
    t = Time('2026-01-01') + day * u.day
    altaz = sc.transform_to(AltAz(obstime=t, location=_LOC))
    return altaz.alt.deg > 20.0


def base_grid(total: int, strat: str) -> List[int]:
    if strat == 'weekly':
        return list(range(0, total, 7))
    if strat == 'biweekly':
        return list(range(0, total, 14))
    if strat == 'monthly':
        return list(range(0, total, 30))
    if strat == 'loglinear':
        return [0, 1, 3, 7, 14, 30, 60, 90] + list(range(120, total, 60))
    if strat == 'adaptive':
        return list(range(0, total, 14))
    raise ValueError(strat)


def build_schedule(pop: List[Pulsar],
                   total_days: int,
                   strategy: str = 'weekly',
                   rng: np.random.Generator | None = None
                   ) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """
    Returns:  schedule[pid]=list(days),  tint[pid]=list(sec)
    """
    rng = rng or np.random.default_rng(123)
    S: Dict[int, List[int]] = {}
    tint: Dict[int, List[float]] = {}
    grid = base_grid(total_days, strategy)

    # nightly beam bins
    nightly: Dict[int, List[int]] = {d: [] for d in grid}
    for p in pop:
        nights = [d for d in grid if _visible(p, d)]
        for d in nights:
            nightly[d].append(p.pid)

    # apply beam limit
    for d, lst in nightly.items():
        if len(lst) > _BEAM_MAX:
            cut = rng.choice(lst, size=len(lst) - _BEAM_MAX, replace=False)
            nightly[d] = [x for x in lst if x not in set(cut)]

    # build per‑pulsar schedule / integration times
    for p in pop:
        days = [d for d in grid if p.pid in nightly.get(d, [])]
        days = [d for d in days if rng.random() > 0.15]  # 15 % random loss
        S[p.pid] = sorted(days)
        if strategy == 'weekly':
            tint[p.pid] = [600.0] * len(days)          # 10 min
        elif strategy == 'biweekly':
            tint[p.pid] = [900.0] * len(days)          # 15 min
        else:
            tint[p.pid] = [1800.0] * len(days)         # 30 min

    # adaptive refinement
    if strategy == 'adaptive':
        from adaptive import refine
        refine(pop, S, tint, total_days // 365, rng)

    return S, tint
