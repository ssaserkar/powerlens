#!/usr/bin/env python3
"""
Paper experiment runner for PowerLens arXiv paper.

Runs all 5 models across 3 power modes with 5 repetitions each.
Extracts per-inference energy (corrected for iterations-per-run).

Usage:
    sudo python experiments/paper_experiments.py 2>&1 | tee experiments/experiment_log.txt

Takes approximately 60-90 minutes.
"""

import subprocess
import json
import time
import re
import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

MODELS = {
    "mobilenetv2": "models/mobilenetv2.onnx",
    "resnet18": "models/resnet18.onnx",
    "resnet34": "models/resnet34.onnx",
    "resnet50": "models/resnet50.onnx",
    "efficientnet_b0": "models/efficientnet_b0.onnx",
}

POWER_MODES = {
    "15W": 0,
    "25W": 1,
    "MAXN_SUPER": 2,
}

# Power mode specs (from nvpmodel --verbose)
MODE_SPECS = {
    "15W": {
        "tdp_w": 15, "cpu_cores": 6, "cpu_max_mhz": 1498,
        "gpu_max_mhz": 612, "emc_max_mhz": 2133,
    },
    "25W": {
        "tdp_w": 25, "cpu_cores": 6, "cpu_max_mhz": 1344,
        "gpu_max_mhz": 918, "emc_max_mhz": 3199,
    },
    "MAXN_SUPER": {
        "tdp_w": 40, "cpu_cores": 6, "cpu_max_mhz": 1728,
        "gpu_max_mhz": 1020, "emc_max_mhz": 3199,
    },
}

NUM_REPETITIONS = 5       # Full profile runs per model-mode combo
RUNS_PER_PROFILE = 50     # PowerLens --runs parameter
STABILIZATION_WAIT = 30   # Seconds after mode switch
OUTPUT_DIR = Path("experiments/results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg):
    """Print with timestamp."""
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def set_power_mode(mode_id, mode_name):
    """Switch power mode and wait for stabilization."""
    log(f"Switching to {mode_name} (mode {mode_id})")

    result = subprocess.run(
        ["sudo", "nvpmodel", "-m", str(mode_id)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr.strip()}")
        return False

    # Verify
    result = subprocess.run(
        ["sudo", "nvpmodel", "-q"],
        capture_output=True, text=True,
    )
    current = result.stdout.strip()
    log(f"  Current: {current}")

    log(f"  Waiting {STABILIZATION_WAIT}s for stabilization...")
    time.sleep(STABILIZATION_WAIT)
    return True


def get_thermal_snapshot():
    """Read all thermal zone temperatures."""
    temps = {}
    thermal_base = Path("/sys/class/thermal")
    for zone in sorted(thermal_base.glob("thermal_zone*")):
        try:
            temp_text = (zone / "temp").read_text(errors="replace").strip()
            temp = int(temp_text) / 1000
            name = (zone / "type").read_text(errors="replace").strip()
            temps[name] = temp
        except (IOError, ValueError, TypeError):
            pass
    return temps


def parse_powerlens_output(output):
    """Parse PowerLens profile output to extract all metrics.
    
    Returns dict with per-inference corrected values.
    """
    result = {}

    # Iterations per run
    m = re.search(r"Using (\d+) iterations per run", output)
    result["iterations_per_run"] = int(m.group(1)) if m else 1

    # Single inference latency (from auto-detect)
    m = re.search(r"Single inference: ([\d.]+) ms", output)
    result["single_inference_ms"] = float(m.group(1)) if m else None

    # Per-inference energy — new report format
    # "Energy/inference:   0.0158 +/- 0.0025 J"
    m = re.search(r"Energy/inference:\s+([\d.]+) \+/- ([\d.]+) J", output)
    if m:
        result["energy_per_inference_j"] = float(m.group(1))
        result["energy_per_inference_std_j"] = float(m.group(2))
    else:
        # Fallback: old format
        m_old = re.search(r"Energy per inference:\s+([\d.]+) J", output)
        if m_old:
            result["energy_per_inference_j"] = float(m_old.group(1))

    # Latency per inference — try new format first, then old
    # "Latency/inference:  1.5 ms"
    m = re.search(r"Latency/inference:\s+([\d.]+) ms", output)
    if m:
        result["latency_per_inference_ms"] = float(m.group(1))
    else:
        # Old format fallback
        m_old = re.search(r"Latency per inference:\s+([\d.]+) ms", output)
        if m_old:
            result["latency_per_inference_ms"] = float(m_old.group(1))
        else:
            # Last resort: use single_inference_ms from auto-detect
            result["latency_per_inference_ms"] = result.get(
                "single_inference_ms"
            )

    # Per-run energy (for reference)
    m = re.search(r"Per-run energy:\s+([\d.]+) \+/- ([\d.]+) J", output)
    if m:
        result["energy_per_run_mean_j"] = float(m.group(1))
        result["energy_per_run_std_j"] = float(m.group(2))

    # Min/Max per inference (already corrected in new report)
    m = re.search(r"Min:\s+([\d.]+) J", output)
    if m:
        result["energy_per_inference_min_j"] = float(m.group(1))

    m = re.search(r"Max:\s+([\d.]+) J", output)
    if m:
        result["energy_per_inference_max_j"] = float(m.group(1))

    # Power stats
    m = re.search(r"Power \(avg\):\s+([\d.]+) W", output)
    result["avg_power_w"] = float(m.group(1)) if m else None

    m = re.search(r"Power \(peak\):\s+([\d.]+) W", output)
    result["peak_power_w"] = float(m.group(1)) if m else None

    m = re.search(r"Power \(idle\):\s+([\d.]+) W", output)
    result["idle_power_w"] = float(m.group(1)) if m else None

    # Efficiency (already corrected in new report)
    m = re.search(r"Efficiency:\s+([\d.]+) inferences/J", output)
    if m:
        result["inferences_per_joule"] = float(m.group(1))

    # Rail breakdown
    rails = {}
    for m in re.finditer(
        r"(VDD_\w+)\s+([\d.]+) W \((\d+)%\)", output
    ):
        rails[m.group(1)] = {
            "power_w": float(m.group(2)),
            "percent": int(m.group(3)),
        }
    result["rails"] = rails

    # Thermal
    thermals = {}
    for m in re.finditer(
        r"(\S+-thermal)\s+avg=([\d.]+)..C\s+max=([\d.]+)..C", output
    ):
        thermals[m.group(1)] = {
            "avg_c": float(m.group(2)),
            "max_c": float(m.group(3)),
        }
    result["thermals"] = thermals

    # Throttling — check for explicit "No" first
    if "No thermal throttling detected" in output:
        result["throttling_detected"] = False
    elif "thermal throttling detected" in output.lower():
        result["throttling_detected"] = True
    else:
        result["throttling_detected"] = None

    # GPU utilization
    m = re.search(
        r"GPU util:\s+avg=(\d+)%\s+max=(\d+)%\s+min=(\d+)%", output
    )
    if m:
        result["gpu_util_avg_pct"] = int(m.group(1))
        result["gpu_util_max_pct"] = int(m.group(2))
        result["gpu_util_min_pct"] = int(m.group(3))

    m = re.search(
        r"GPU freq:\s+avg=(\d+)MHz\s+max=(\d+)MHz", output
    )
    if m:
        result["gpu_freq_avg_mhz"] = int(m.group(1))
        result["gpu_freq_max_mhz"] = int(m.group(2))

    # Sample rate
    m = re.search(r"Sample rate:\s+([\d.]+) Hz", output)
    result["sample_rate_hz"] = float(m.group(1)) if m else None

    return result


def run_single_profile(model_name, model_path):
    """Run one PowerLens profile and parse results."""
    result = subprocess.run(
        [
            sys.executable, "-m", "powerlens", "profile",
            "--onnx", model_path,
            "--runs", str(RUNS_PER_PROFILE),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        log(f"    ERROR: {result.stderr[:300]}")
        return None

    parsed = parse_powerlens_output(result.stdout)
    parsed["raw_stdout"] = result.stdout
    return parsed


def run_experiment_matrix():
    """Run all models × all power modes × repetitions."""
    all_results = []

    for mode_name, mode_id in POWER_MODES.items():
        log(f"\n{'='*60}")
        log(f"POWER MODE: {mode_name}")
        log(f"{'='*60}")

        if not set_power_mode(mode_id, mode_name):
            log(f"Failed to set mode {mode_name}, skipping")
            continue

        for model_name, model_path in MODELS.items():
            if not Path(model_path).exists():
                log(f"  SKIP: {model_path} not found")
                continue

            log(f"\n  Model: {model_name} @ {mode_name}")

            for rep in range(NUM_REPETITIONS):
                log(f"    Repetition {rep + 1}/{NUM_REPETITIONS}")

                parsed = run_single_profile(model_name, model_path)

                if parsed:
                    entry = {
                        "model": model_name,
                        "power_mode": mode_name,
                        "mode_id": mode_id,
                        "repetition": rep,
                        "mode_specs": MODE_SPECS[mode_name],
                        **{k: v for k, v in parsed.items()
                           if k != "raw_stdout"},
                    }
                    all_results.append(entry)

                    # Print key metric
                    epi = parsed.get("energy_per_inference_j")
                    lat = parsed.get("latency_per_inference_ms")
                    pwr = parsed.get("avg_power_w")
                    if epi is not None:
                        log(f"      → {epi*1000:.1f} mJ/inf, "
                            f"{lat or 0:.1f} ms, {pwr or 0:.1f} W")
                    else:
                        log(f"      → Parse failed, raw output saved")

                # Cool down between repetitions
                time.sleep(5)

            # Cool down between models
            log(f"  Cooling down 15s between models...")
            time.sleep(15)

    return all_results

def run_tegrastats_validation():
    """Run tegrastats alongside PowerLens for validation."""
    log("\n" + "=" * 60)
    log("VALIDATION: tegrastats comparison")
    log("=" * 60)

    # Make sure we're in MAXN
    set_power_mode(2, "MAXN_SUPER")

    tegra_log = OUTPUT_DIR / f"tegrastats_{TIMESTAMP}.log"

    # Kill any existing tegrastats
    subprocess.run(
        ["killall", "tegrastats"],
        capture_output=True,
    )
    time.sleep(1)

    # Start tegrastats (we're already running as sudo)
    log("Starting tegrastats...")
    tegra_proc = subprocess.Popen(
        ["tegrastats", "--interval", "100",
         "--logfile", str(tegra_log)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(3)  # Let tegrastats start

    # Run PowerLens
    log("Running PowerLens profile (ResNet-18, 100 runs)...")
    result = subprocess.run(
        [
            sys.executable, "-m", "powerlens", "profile",
            "--onnx", MODELS["resnet18"],
            "--runs", "100",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    time.sleep(3)

    # Stop tegrastats
    log("Stopping tegrastats...")
    tegra_proc.terminate()
    try:
        tegra_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tegra_proc.kill()
        tegra_proc.wait()

    # Also kill by name in case
    subprocess.run(
        ["killall", "tegrastats"],
        capture_output=True,
    )

    powerlens_parsed = None
    if result.returncode == 0:
        powerlens_parsed = parse_powerlens_output(result.stdout)
        log(f"  PowerLens profile completed successfully")
    else:
        log(f"  PowerLens ERROR: {result.stderr[:200]}")

    # Parse tegrastats log
    tegra_powers = []
    if tegra_log.exists():
        with open(tegra_log, errors="replace") as f:
            for line in f:
                # tegrastats format varies by JetPack version
                # Try: VDD_IN 8000mW/8000mW
                m = re.search(r"VDD_IN\s+(\d+)mW/(\d+)mW", line)
                if m:
                    tegra_powers.append({
                        "current_mw": int(m.group(1)),
                        "average_mw": int(m.group(2)),
                    })
                else:
                    # Try alternate format: VDD_IN 8000/8000
                    m = re.search(r"VDD_IN\s+(\d+)/(\d+)", line)
                    if m:
                        tegra_powers.append({
                            "current_mw": int(m.group(1)),
                            "average_mw": int(m.group(2)),
                        })
        log(f"  tegrastats samples: {len(tegra_powers)}")
    else:
        log(f"  WARNING: tegrastats log not found at {tegra_log}")

    validation = {
        "tegrastats_log": str(tegra_log),
        "tegrastats_samples": len(tegra_powers),
        "powerlens_result": (
            {k: v for k, v in powerlens_parsed.items()
             if k != "raw_stdout"}
            if powerlens_parsed else None
        ),
    }

    if tegra_powers:
        avg_current = sum(
            t["current_mw"] for t in tegra_powers
        ) / len(tegra_powers)
        validation["tegrastats_avg_power_mw"] = avg_current
        validation["tegrastats_avg_power_w"] = avg_current / 1000

        if powerlens_parsed and powerlens_parsed.get("rails"):
            pl_vdd_in = powerlens_parsed["rails"].get(
                "VDD_IN", {}
            ).get("power_w", 0)
            if pl_vdd_in > 0:
                error_pct = abs(
                    pl_vdd_in - avg_current / 1000
                ) / (avg_current / 1000) * 100
                validation["vdd_in_error_pct"] = error_pct
                log(f"  tegrastats VDD_IN: {avg_current / 1000:.2f} W")
                log(f"  PowerLens  VDD_IN: {pl_vdd_in:.2f} W")
                log(f"  Error: {error_pct:.1f}%")
    else:
        log("  WARNING: No tegrastats power data parsed")
        log("  Checking tegrastats log format...")
        if tegra_log.exists():
            with open(tegra_log, errors="replace") as f:
                first_lines = [f.readline() for _ in range(3)]
            for line in first_lines:
                log(f"    {line.strip()[:100]}")

    return validation


def run_thermal_stress_test():
    """Run sustained inference for thermal characterization."""
    log("\n" + "=" * 60)
    log("THERMAL: Sustained inference stress test")
    log("=" * 60)

    set_power_mode(2, "MAXN_SUPER")

    # Record initial temps after cooldown
    log("  Waiting 30s for thermal baseline...")
    time.sleep(30)

    initial_temps = get_thermal_snapshot()
    log(f"  Initial GPU temp: "
        f"{initial_temps.get('gpu-thermal', 'N/A')}°C")
    log(f"  Initial CPU temp: "
        f"{initial_temps.get('cpu-thermal', 'N/A')}°C")

    # Sample temps during the run
    temp_timeline = []
    temp_timeline.append({
        "time_s": 0,
        "temps": initial_temps,
    })

    # Run sustained inference in chunks to capture temp timeline
    log("  Running sustained inference (5 chunks × 40 runs)...")
    total_start = time.monotonic()

    all_profile_results = []
    for chunk in range(5):
        chunk_start = time.monotonic()

        result = subprocess.run(
            [
                sys.executable, "-m", "powerlens", "profile",
                "--onnx", MODELS["resnet50"],
                "--runs", "40",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        elapsed = time.monotonic() - total_start
        current_temps = get_thermal_snapshot()
        temp_timeline.append({
            "time_s": round(elapsed, 1),
            "temps": current_temps,
        })

        if result.returncode == 0:
            parsed = parse_powerlens_output(result.stdout)
            all_profile_results.append(parsed)

        gpu_temp = current_temps.get("gpu-thermal", 0)
        log(f"    Chunk {chunk + 1}/5: "
            f"GPU={gpu_temp:.1f}°C "
            f"(t={elapsed:.0f}s)")

    # Final temps
    final_temps = get_thermal_snapshot()
    total_elapsed = time.monotonic() - total_start

    log(f"\n  Test duration: {total_elapsed:.0f}s")
    log(f"  Final GPU temp: "
        f"{final_temps.get('gpu-thermal', 'N/A')}°C")
    log(f"  Temp rise: "
        f"{final_temps.get('gpu-thermal', 0) - initial_temps.get('gpu-thermal', 0):.1f}°C")

    thermal_result = {
        "duration_s": round(total_elapsed, 1),
        "initial_temps": initial_temps,
        "final_temps": final_temps,
        "temp_rise": {
            k: round(
                final_temps.get(k, 0) - initial_temps.get(k, 0), 1
            )
            for k in initial_temps
            if k in final_temps
        },
        "temp_timeline": temp_timeline,
        "num_chunks": 5,
        "runs_per_chunk": 40,
        "model": "resnet50",
        "power_mode": "MAXN_SUPER",
        "chunk_results": [
            {k: v for k, v in p.items() if k != "raw_stdout"}
            for p in all_profile_results
        ],
    }

    # Check for throttling
    max_gpu_temp = max(
        t["temps"].get("gpu-thermal", 0)
        for t in temp_timeline
    )
    thermal_result["max_gpu_temp_c"] = max_gpu_temp
    thermal_result["throttle_threshold_c"] = 85.0
    thermal_result["throttled"] = max_gpu_temp >= 85.0

    log(f"  Max GPU temp: {max_gpu_temp:.1f}°C "
        f"(threshold: 85°C)")
    log(f"  Throttled: {'YES' if max_gpu_temp >= 85 else 'No'}")

    return thermal_result


def compute_summary_tables(all_results):
    """Compute summary statistics for the paper tables."""
    import statistics

    log("\n" + "="*60)
    log("SUMMARY TABLES FOR PAPER")
    log("="*60)

    # Table: Per-inference energy at MAXN
    log("\n--- Table: Per-Inference Energy (MAXN_SUPER) ---")
    log(f"{'Model':20s} {'Energy(mJ)':>10s} {'±(mJ)':>8s} "
        f"{'Lat(ms)':>8s} {'Power(W)':>8s} {'GPU%':>6s} "
        f"{'GPUFreq':>8s}")
    log("-" * 75)

    summary = {}
    for mode_name in POWER_MODES:
        summary[mode_name] = {}
        for model_name in MODELS:
            runs = [
                r for r in all_results
                if r["model"] == model_name
                and r["power_mode"] == mode_name
                and r.get("energy_per_inference_j") is not None
            ]
            if not runs:
                continue

            energies = [r["energy_per_inference_j"] * 1000 for r in runs]  # mJ
            latencies = [
                r["latency_per_inference_ms"]
                for r in runs
                if r.get("latency_per_inference_ms")
            ]
            powers = [
                r["avg_power_w"]
                for r in runs
                if r.get("avg_power_w")
            ]
            gpu_utils = [
                r["gpu_util_avg_pct"]
                for r in runs
                if r.get("gpu_util_avg_pct")
            ]
            gpu_freqs = [
                r["gpu_freq_avg_mhz"]
                for r in runs
                if r.get("gpu_freq_avg_mhz")
            ]

            entry = {
                "energy_mj_mean": statistics.mean(energies),
                "energy_mj_std": (
                    statistics.stdev(energies) if len(energies) > 1 else 0
                ),
                "latency_ms_mean": (
                    statistics.mean(latencies) if latencies else 0
                ),
                "avg_power_w_mean": (
                    statistics.mean(powers) if powers else 0
                ),
                "gpu_util_avg": (
                    statistics.mean(gpu_utils) if gpu_utils else 0
                ),
                "gpu_freq_avg": (
                    statistics.mean(gpu_freqs) if gpu_freqs else 0
                ),
                "num_runs": len(runs),
                "inferences_per_joule": (
                    statistics.mean([
                        r["inferences_per_joule"]
                        for r in runs
                        if r.get("inferences_per_joule")
                    ]) if any(r.get("inferences_per_joule") for r in runs)
                    else (1000 / statistics.mean(energies)
                          if statistics.mean(energies) > 0 else 0)
                ),
            }
            summary[mode_name][model_name] = entry

            # Also extract per-rail averages
            rail_powers = {}
            for r in runs:
                if r.get("rails"):
                    for rail_name, rail_data in r["rails"].items():
                        if rail_name not in rail_powers:
                            rail_powers[rail_name] = []
                        rail_powers[rail_name].append(
                            rail_data["power_w"]
                        )
            entry["rails"] = {
                name: statistics.mean(vals)
                for name, vals in rail_powers.items()
            }

    # Print MAXN table
    for model_name in MODELS:
        if model_name in summary.get("MAXN_SUPER", {}):
            e = summary["MAXN_SUPER"][model_name]
            log(f"{model_name:20s} {e['energy_mj_mean']:>10.1f} "
                f"{e['energy_mj_std']:>8.1f} "
                f"{e['latency_ms_mean']:>8.1f} "
                f"{e['avg_power_w_mean']:>8.1f} "
                f"{e['gpu_util_avg']:>6.0f} "
                f"{e['gpu_freq_avg']:>8.0f}")

    # Table: Efficiency across power modes
    log("\n--- Table: Efficiency Across Power Modes ---")
    log(f"{'Model':20s} {'15W inf/J':>10s} {'25W inf/J':>10s} "
        f"{'MAXN inf/J':>10s} {'Best':>8s}")
    log("-" * 65)

    for model_name in MODELS:
        vals = {}
        for mode_name in POWER_MODES:
            if model_name in summary.get(mode_name, {}):
                vals[mode_name] = (
                    summary[mode_name][model_name]["inferences_per_joule"]
                )

        if vals:
            best = max(vals, key=vals.get)
            log(f"{model_name:20s} "
                f"{vals.get('15W', 0):>10.1f} "
                f"{vals.get('25W', 0):>10.1f} "
                f"{vals.get('MAXN_SUPER', 0):>10.1f} "
                f"{best:>8s}")

    # Table: Per-rail breakdown at MAXN
    log("\n--- Table: Per-Rail Power Breakdown (MAXN_SUPER) ---")
    log(f"{'Model':20s} {'VDD_IN(W)':>10s} {'CPU_GPU(W)':>10s} "
        f"{'SOC(W)':>10s} {'SOC%':>6s}")
    log("-" * 60)

    for model_name in MODELS:
        if model_name in summary.get("MAXN_SUPER", {}):
            e = summary["MAXN_SUPER"][model_name]
            rails = e.get("rails", {})
            vdd_in = rails.get("VDD_IN", 0)
            cpu_gpu = rails.get("VDD_CPU_GPU_CV", 0)
            soc = rails.get("VDD_SOC", 0)
            total = cpu_gpu + soc
            soc_pct = (soc / total * 100) if total > 0 else 0
            log(f"{model_name:20s} {vdd_in:>10.2f} {cpu_gpu:>10.2f} "
                f"{soc:>10.2f} {soc_pct:>6.1f}")

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we should skip experiment 1 (already done)
    skip_matrix = "--skip-matrix" in sys.argv

    log("="*60)
    log("PowerLens Paper Experiment Suite")
    log(f"Timestamp: {TIMESTAMP}")
    log("="*60)

    # Check models exist
    missing = [
        name for name, path in MODELS.items()
        if not Path(path).exists()
    ]
    if missing:
        log(f"ERROR: Missing models: {missing}")
        log("Run: python experiments/export_models.py")
        sys.exit(1)

    log(f"Models found: {list(MODELS.keys())}")
    log(f"Power modes: {list(POWER_MODES.keys())}")
    log(f"Repetitions per combo: {NUM_REPETITIONS}")
    log(f"Total profiles: {len(MODELS) * len(POWER_MODES) * NUM_REPETITIONS}")
    log(f"Estimated time: ~60-90 minutes")

    # Record ambient conditions
    ambient = get_thermal_snapshot()
    log(f"\nAmbient temps: gpu={ambient.get('gpu-thermal', '?')}°C, "
        f"cpu={ambient.get('cpu-thermal', '?')}°C")

    results = {
        "timestamp": TIMESTAMP,
        "config": {
            "models": list(MODELS.keys()),
            "power_modes": POWER_MODES,
            "mode_specs": MODE_SPECS,
            "num_repetitions": NUM_REPETITIONS,
            "runs_per_profile": RUNS_PER_PROFILE,
        },
        "ambient_temps": ambient,
    }

    # ========================================================
    # EXPERIMENT 1: Model × Power Mode Matrix
    # ========================================================
    if skip_matrix:
        log("\nSkipping experiment 1 (--skip-matrix flag)")
        # Load from interim file
        interim_files = sorted(OUTPUT_DIR.glob("interim_*.json"))
        if interim_files:
            with open(interim_files[-1]) as f:
                interim = json.load(f)
            all_results = interim.get("experiment_matrix", [])
            log(f"Loaded {len(all_results)} results from {interim_files[-1]}")
        else:
            all_results = []
            log("WARNING: No interim results found")
    else:
        log("\n" + "#"*60)
        log("# EXPERIMENT 1: Model × Power Mode Matrix")
        log(f"# {len(MODELS)} models × {len(POWER_MODES)} modes × "
            f"{NUM_REPETITIONS} reps = "
            f"{len(MODELS) * len(POWER_MODES) * NUM_REPETITIONS} profiles")
        log("#"*60)

        all_results = run_experiment_matrix()

        # Save intermediate results
        interim_file = OUTPUT_DIR / f"interim_{TIMESTAMP}.json"
        with open(interim_file, "w") as f:
            json.dump({**results, "experiment_matrix": all_results},
                      f, indent=2, default=str)
        log(f"\nInterim results saved: {interim_file}")

    results["experiment_matrix"] = all_results

    # Save intermediate results
    interim_file = OUTPUT_DIR / f"interim_{TIMESTAMP}.json"
    with open(interim_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nInterim results saved: {interim_file}")

    # ========================================================
    # EXPERIMENT 2: Tegrastats Validation
    # ========================================================
    log("\n" + "#"*60)
    log("# EXPERIMENT 2: Tegrastats Validation")
    log("#"*60)

    validation = run_tegrastats_validation()
    results["validation"] = validation

    # ========================================================
    # EXPERIMENT 3: Thermal Stress Test
    # ========================================================
    log("\n" + "#"*60)
    log("# EXPERIMENT 3: Thermal Stress Test")
    log("#"*60)

    thermal = run_thermal_stress_test()
    results["thermal_stress"] = thermal

    # ========================================================
    # COMPUTE SUMMARY TABLES
    # ========================================================
    summary = compute_summary_tables(all_results)
    results["summary"] = summary

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================
    output_file = OUTPUT_DIR / f"paper_results_{TIMESTAMP}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log("\n" + "="*60)
    log("ALL EXPERIMENTS COMPLETE")
    log(f"Results: {output_file}")
    log("="*60)

    return results


if __name__ == "__main__":
    main()