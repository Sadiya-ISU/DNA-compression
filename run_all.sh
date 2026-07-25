#!/usr/bin/env bash
# Reproduce every artefact in the report from a fresh clone.
#
# Idempotent: skips downloads that are cached, regenerates everything else.
# Exits non-zero if anything fails.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== Step 1/5: install dev dependencies ==="
python3 -m pip install --user -q -r requirements.txt

echo
echo "=== Step 2/5: fetch real reference genomes (best effort) ==="
python3 fetch_real_genomes.py || echo "(real genomes unavailable; proceeding with synthetic only)"

echo
echo "=== Step 3/5: run test suite ==="
python3 -m pytest tests/ -v --tb=short

echo
echo "=== Step 4/5: run full benchmark + reports + plots ==="
python3 comparison.py --full-pipeline

echo
echo "=== Step 5/5: run VCSD+ ablation ==="
python3 ablation.py

echo
echo "=== Done. Artifacts under Results/ and docs/. ==="
ls -la Results/
