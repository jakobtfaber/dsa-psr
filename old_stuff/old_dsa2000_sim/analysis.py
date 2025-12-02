"""
analysis.py  –  compute metrics & arrays for plotting
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from population import Pulsar
from simulate import simulate_residuals


def lomb_detect(res, thresh=7.0):
    power = np.abs(np.fft.rfft(res)) ** 2
    return power.max() > thresh * np.median(power)


def fit_spin(p: Pulsar, days, res):
    t = np.array(days) * 86400.0
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, res, rcond=None)
    return coef[1] / p.P         # estimate ΔṖ


def assess(pop: List[Pulsar],
           sched: Dict[int, List[int]],
           tint: Dict[int, List[float]],
           years: int,
           rng):
    phase_ok, pd_err, bin_detect = [], [], []

    # arrays for survival curve
    yearly_bins = np.arange(0, years + 0.1, 0.25)  # 3‑month steps
    surv = np.ones_like(yearly_bins, dtype=float)

    for p in pop:
        days = sched[p.pid]
        if not days:
            phase_ok.append(False)
            continue
        res = simulate_residuals(p, days, tint[p.pid], rng)

        # phase loss
        ok = not np.any(np.abs(res) > 0.8 * p.P)
        phase_ok.append(ok)
        if ok:
            # survival curve update
            last_year = days[-1] / 365
            surv[yearly_bins <= last_year] += 1

        # ΔṖ error
        est = fit_spin(p, days, res)
        pd_err.append(abs(est / p.Pdot))

        # binary detection
        if p.is_binary:
            bin_detect.append(lomb_detect(res))

    phase_frac = np.mean(phase_ok)
    med_pd_err = np.median(pd_err)
    bin_rate = np.mean(bin_detect) if bin_detect else 0.0

    # normalise survival (first point = total pop)
    surv /= len(pop)

    return phase_frac, bin_rate, med_pd_err, \
        np.column_stack([yearly_bins, surv]), \
        np.array(bin_detect, dtype=int), \
        np.array(pd_err)
