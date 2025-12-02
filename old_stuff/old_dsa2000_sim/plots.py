"""
plots.py – regenerate figures (PDF) from saved .npy arrays
"""

import json, pathlib, numpy as np, matplotlib.pyplot as plt, seaborn as sns
sns.set_context('paper')

COL = dict(weekly='tab:green', biweekly='tab:blue',
            monthly='tab:red', loglinear='tab:orange',
            adaptive='tab:purple')

def _load(strategy, name):
    f = pathlib.Path('results') / strategy / f'{name}.npy'
    return np.load(f) if f.exists() else None


def phase_survival():
    plt.figure(figsize=(4, 3))
    for s, c in COL.items():
        arr = _load(s, 'phase_curve')
        if arr is not None:
            plt.plot(arr[:, 0], arr[:, 1], label=s, color=c)
    plt.xlabel('Year')
    plt.ylabel('Fraction phase‑connected')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/phase_survival.pdf')


def binary_vs_pb():
    plt.figure(figsize=(4, 3))
    for s, c in COL.items():
        arr = _load(s, 'bin_curve')
        if arr is not None:
            plt.step(arr[:, 0], arr[:, 1], where='mid', label=s, color=c)
    plt.xscale('log')
    plt.xlabel('Orbital period $P_b$ [d]')
    plt.ylabel('Detection fraction')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/binary_vs_Pb.pdf')


def pdot_hist():
    plt.figure(figsize=(4, 3))
    bins = np.logspace(-3, 1, 40)
    for s, c in COL.items():
        arr = _load(s, 'pdot_err')
        if arr is not None:
            sns.histplot(arr, bins=bins, element='step',
                         stat='density', label=s, color=c, alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Fractional $\\dot P$ error')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/pdot_error.pdf')


def cadence_heat():
    plt.figure(figsize=(5, 3))
    for s, c in COL.items():
        if s not in ('weekly', 'biweekly', 'monthly'):
            continue
        arr = _load(s, 'visits')
        if arr is not None:
            plt.scatter(arr[:, 0], arr[:, 1], s=4, alpha=0.3, color=c, label=s)
    plt.xlabel('Mean visits / pulsar / year')
    plt.ylabel('Mean $t_\\mathrm{int}$ [min]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/cadence_heat.pdf')


if __name__ == '__main__':
    phase_survival()
    binary_vs_pb()
    pdot_hist()
    cadence_heat()
    print('Figures written to results/*.pdf')
