"""
noise.py  –  coloured‑noise generator
"""

import numpy as np


def powerlaw_red_noise(times, amp, gamma, rng):
    """
    FFT‑based red noise with PSD ∝ f^(‑gamma).
    Zero frequency component is set to zero to avoid divergence.
    """
    if amp == 0 or gamma == 0:
        return np.zeros_like(times, dtype=float)

    n = len(times)
    dt = times[1] - times[0]
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = np.where(freqs == 0, 0.0, amp * (freqs / 1.0) ** (-gamma))

    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    coeffs = np.sqrt(psd / 2) * (np.cos(phases) + 1j * np.sin(phases))
    series = np.fft.irfft(coeffs, n=n)
    return series - series.mean()
