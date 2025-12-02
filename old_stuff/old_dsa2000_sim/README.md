
# DSA‑2000 Pulsar Timing Campaign Simulation

**Generated:** 2025-05-27

A fully‑reproducible Python package that recreates the results in  
*“Pulsar Timing Campaign Simulation with DSA‑2000 Specifications”*.

The code is modular:

| module | purpose |
|--------|---------|
| `population.py` | generate a synthetic pulsar catalogue (30 000 sources, Galactic‑plane biased, 1 % binaries) |
| `noise.py`      | assign white, red, and DM noise parameters |
| `schedule.py`   | build observing schedules (weekly, bi‑weekly, monthly, log‑linear, adaptive) under DSA‑2000 constraints |
| `simulate.py`   | inject noise + binary delays, produce TOAs/residuals |
| `analysis.py`   | evaluate phase‑connection, binary‑detection, −dot P uncertainty |
| `run_campaign.py` | CLI wrapper to run an end‑to‑end Monte‑Carlo |

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the default 5‑year weekly simulation
python run_campaign.py --strategy weekly --years 5 --n_pulsars 30000
```

Output (per strategy) is a JSON summary plus CSV tables and PDF plots in `results/<strategy>/`.

To reproduce **all** figures in the report (may take several hours):

```bash
./reproduce_report.sh
```

See comments in each module for details and extension points.

---  
*Contact:* ChatGPT · v0.1
