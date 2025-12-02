# Objective
Design an optimal observational cadence strategy for achieving phase-connected timing solutions of newly discovered radio pulsars using a steerable, interferometric radio telescope array. The objective is to maximize timing completeness per source population and optimize telescope time use across a diverse pulsar census in the Milky Way, using analytic reasoning and supporting simulations. Consider two survey strategies: (1) A targeted search in (nominally) 4000 beams steered towards continuum sources; (2) A blind search in (nominally) 1000000 beams that tile the central part of the primary beam. Assume that search-mode data will not be stored, and that searching will need to be done in quasi-real time, on the timescale of the dwell (pointing) time. The telescope backend will run on two NVIDIA Vera Rubin NVL144 GPU racks designed for high-performance computing. Consequently, the dwell time is bounded strongly by memory requirements, but less strongly by the compute requirements.

# Instructions
- Specify the number and spacing of follow-up observations (epochs) needed to ensure phase-connected timing per source, per sub-population.
- Quantify success rates, expected uncertainties (period, period derivative), and justify recommendations analytically.
- Support strategy with detailed simulations for each sub-population.
- Explicitly document all relevant telescope and noise parameters.
- Validate the outputs of each major simulation or calculation.

## Sub-categories
Cadence recommendations should cover:
  1. Young, canonical disk pulsars
  2. Recycled millisecond pulsars (field + globular cluster)
  3. Magnetars
  4. High-B/transition objects (high-B radio pulsars, XDINS)
  5. Emission-intermittent classes (RRATs, nullers, intermittents)
  6. Pulsars in special locales (Galactic-center)
- Incorporate population synthesis fractions from (e.g.) psrpoppy.
- Explicitly include noise contributors: radiometer/template, jitter, ISM/DM/scattering, red noise, glitches, instrumental, array-level, clock, Solar-system ephemeris, and GWB.
- Reference the provided telescope parameters table, fiducial pulsar search parameters table, and NVIDIA Vera Rubin NVL144 specifications table. Define all variables and units.

# Context
Scope: Only newly discovered radio pulsars detected by the specified telescope.
In-scope: Sub-population differentiation, cadence scheduling, noise accounting, simulated and analytic prediction.
Out-of-scope: Broader survey planning, instrument commissioning, non-radio sources.

## Key telescope/reference parameters
| Parameter                     | Value                                          |
|-------------------------------|------------------------------------------------|
| Number of antennas            | 1650                                           |
| Dish diameter                 | 6.15 m                                         |
| Field of view                 | 7 deg² at 1.35 GHz                             |
| Array size                    | 15 km EW × 19 km NS                            |
| Array configuration           | Pseudo-random, minimize PSF sidelobes          |
| PSF                           | 3.05" at 1.35 GHz, circular at DEC=0°          |
| Tsys/eta                      | 25–40 K, elevation dependent                   |
| Array SEFD at boresight       | 1.41–2.25 Jy, elevation dependent              |
| Band                          | 0.7–2 GHz                                      |
| Channelization                | 10⁴ × 130.2 kHz                                |
| Survey tiling                 | 6×5 21-min pointings, ≈2.6 deg² unique FoV per pointing|

## Fiducial parameters for pulsar searches in both targeted and blind search modes
| Parameter              | Targeted Search       | Blind Search             |
|------------------------|-----------------------|--------------------------|
| Number of beams        | 4000                  | 2e5                      |
| Channels               | 2500                  | 2500                     |
| Band                   | Approx. bottom 25%    | Approx. bottom 25%       |
| DM trials              | 500                   | 500                      |
| Time resolution (ms)   | 0.1                   | 0.1                      |
| Maximum dwell time (s) | 1260                  | 60                       |
| Period range (ms to s) | 1 ms – 5 s            | 1 ms – 5 s               |
| Acceleration trials    | 500                   | 5                        |
| Polarizations          | 1                     | 1                        |
| Number of bits         | 4                     | 4                        |

## NVIDIA Vera Rubin NVL144 features/specifications (for one rack)
| Feature                |  Specification                  |
|------------------------|---------------------------------|
| Compute performance    | 50 PFLOPs dense FP4 compute     |
| Memory Configuration   | 288 GB HBM4 per package         |
| Memory Bandwidth       | 13 TB/s aggregate               |
| Network Speed          | NVLink 6: 3.6TB/s bidirectional |
| Power Consumption      | ~1800 W estimated               |

# Reasoning Steps
Decompose the cadence problem per sub-population and noise source. Develop analytic models and match with simulation to validate rates and uncertainties.

# Planning and Verification
- Assess sub-population characteristics (e.g., period, Pdot, noise, abundance).
- For each, develop cadence scenarios, simulate, and compare analytic and simulated outputs.
- Verify consistency with telescope constraints and noise models.
- Optimize for minimum telescope time per source and overall timing yield.
- Ensure simulations and analytic calculations are linked and clearly explained.

