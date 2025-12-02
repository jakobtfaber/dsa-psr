"""
population.py  –  create the synthetic 30 000‑pulsar catalogue
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import numpy as np


@dataclass
class Pulsar:
    pid: int
    P: float               # spin period [s]
    Pdot: float            # period derivative
    l: float               # Galactic longitude [deg]
    b: float               # Galactic latitude  [deg]
    DM: float              # dispersion measure [pc cm^-3]
    is_binary: bool
    Pb: float | None       # orbital period [d]
    ecc: float | None
    x: float | None        # proj. semi‑major axis [lt‑s]
    sigma_w: float         # white‑noise rms [s]
    sigma_r: float         # red‑noise amplitude
    gamma_r: float         # red‑noise index
    dm_amp: float          # DM‑variation amplitude


def generate_population(N: int = 30_000,
                        binary_fraction: float = 0.01,
                        rng: np.random.Generator | None = None
                        ) -> List[Pulsar]:
    """Return a list of Pulsar objects."""
    rng = rng or np.random.default_rng(42)
    logP = rng.normal(-0.3, 0.6, N)            # median P ~0.5 s
    P = 10 ** logP
    logPdot = -13 + 3 * (logP - np.log10(1.0)) # crude P‑Pdot trend
    Pdot = 10 ** logPdot
    l = rng.uniform(0, 360, N)
    b = np.arcsin(rng.uniform(-1, 1, N)) * 180 / np.pi * 0.2   # thin disk
    DM = np.abs(rng.normal(50, 30, N)) + 1

    pop: List[Pulsar] = []
    for i in range(N):
        is_bin = rng.random() < binary_fraction
        if is_bin:
            Pb = 10 ** rng.uniform(-1, 2)          # 0.1–100 d
            ecc = 0.0 if Pb < 1 else rng.uniform(0, 0.9)
            x = 10 ** rng.uniform(-4, -1)          # 0.1–10 lt‑s
        else:
            Pb = ecc = x = None

        sigma_w = 1e-6 * 10 ** rng.normal(0, 0.6)  # 1 µs–0.1 ms

        if rng.random() < 0.5:
            sigma_r = 10 ** rng.uniform(-14, -12)
            gamma_r = rng.uniform(2, 6)
        else:
            sigma_r = gamma_r = 0.0

        dm_amp = rng.uniform(1e-4, 1e-3)

        pop.append(Pulsar(i, float(P[i]), float(Pdot[i]),
                          float(l[i]), float(b[i]), float(DM[i]),
                          is_bin, Pb, ecc, x,
                          sigma_w, sigma_r, gamma_r, dm_amp))
    return pop


# helper for notebooks
def as_dataframe(pop):
    import pandas as pd
    return pd.DataFrame([asdict(p) for p in pop])
