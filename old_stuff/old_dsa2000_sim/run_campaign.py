#!/usr/bin/env python
"""
run_campaign.py – one‑shot Monte‑Carlo driver
"""

from __future__ import annotations
import argparse, json, pathlib, numpy as np
from population import generate_population
from schedule import build_schedule
from analysis import assess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy',
                    choices=['weekly', 'biweekly', 'monthly',
                             'loglinear', 'adaptive'],
                    default='weekly')
    ap.add_argument('--years', type=int, default=5)
    ap.add_argument('--n_pulsars', type=int, default=30_000)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    pop = generate_population(args.n_pulsars, rng=rng)
    sched, tint = build_schedule(pop, args.years * 365,
                                 strategy=args.strategy, rng=rng)

    phase, binrate, pd_err, curve, bins, pderr_arr = \
        assess(pop, sched, tint, args.years, rng)

    outdir = pathlib.Path('results') / args.strategy
    outdir.mkdir(parents=True, exist_ok=True)

    json.dump({'strategy': args.strategy,
               'phase_connected_fraction': round(float(phase), 3),
               'binary_detection_fraction': round(float(binrate), 3),
               'median_frac_Pdot_error': round(float(pd_err), 3)},
              open(outdir / 'summary.json', 'w'), indent=2)

    np.save(outdir / 'phase_curve.npy', curve)
    # binary curve vs log Pb (for plotting convenience)
    np.save(outdir / 'bin_curve.npy',
            np.column_stack([np.linspace(0.1, 100, 25),
                             np.cumsum(bins) / bins.sum()]))
    np.save(outdir / 'pdot_err.npy', pderr_arr)

    # visits for cadence heatmap
    visits_per = np.array([len(sched[pid]) / args.years for pid in sched])
    t_mean = np.array([np.mean(tint[pid]) / 60 for pid in tint])
    np.save(outdir / 'visits.npy',
            np.column_stack([visits_per, t_mean]))

    print((outdir / 'summary.json').read_text())


if __name__ == '__main__':
    main()
