#!/usr/bin/env python3
"""Run only tegrastats validation and thermal stress test."""
import json
import sys
sys.path.insert(0, ".")

# Import from the main experiment script
from experiments.paper_experiments import (
    run_tegrastats_validation,
    run_thermal_stress_test,
    log, OUTPUT_DIR, TIMESTAMP,
)
from pathlib import Path

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load interim results
interim_files = sorted(OUTPUT_DIR.glob("interim_*.json"))
if interim_files:
    with open(interim_files[-1]) as f:
        results = json.load(f)
    log(f"Loaded interim results: {interim_files[-1]}")
else:
    log("No interim results found, starting fresh")
    results = {}

# Run experiment 2
log("Running tegrastats validation...")
results["validation"] = run_tegrastats_validation()

# Run experiment 3
log("Running thermal stress test...")
results["thermal_stress"] = run_thermal_stress_test()

# Save
output_file = OUTPUT_DIR / f"paper_results_{TIMESTAMP}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=str)

log(f"Results saved: {output_file}")