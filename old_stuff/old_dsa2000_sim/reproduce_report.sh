#!/usr/bin/env bash
set -e
for strat in weekly biweekly monthly loglinear adaptive; do
  echo "=== $strat ==="
  python run_campaign.py --strategy "$strat" --years 5 --n_pulsars 30000
done
python plots.py
