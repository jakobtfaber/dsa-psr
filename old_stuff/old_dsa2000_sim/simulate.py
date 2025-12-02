"""
simulate.py  –  inject noise + orbital Roemer delay
"""

from __future__ import annotations
import numpy as np
from typing import List
from population import Pulsar
from noise import powerlaw_red_noise

T_REF = 1800.0  # reference integration time 30 min [s]


def simulate_residuals(p: Pulsar,
                       days: List[int],
                       t_int: List[float],
                       rng: np.random.Generator):
    """
    Return array of timing residuals [s] for one pulsar.
    """
    wn_scale = np.sqrt(T_REF / np.array(t_int))
    res = rng.normal(0, p.sigma_w * wn_scale)

    # red noise
    t = np.array(days) * 86400.0
    res += powerlaw_red_noise(t, p.sigma_r, p.gamma_r, rng)

    # DM annual wave (simplified)
    res += 4.15e-3 * p.dm_amp * np.sin(2 * np.pi * np.array(days) / 365)

    # binary Roemer delay
    if p.is_binary:
        res += p.x * np.sin(2 * np.pi * np.array(days) / p.Pb)

    return res
