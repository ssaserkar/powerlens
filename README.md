# ⚡ PowerLens

[![Tests](https://github.com/ssaserkar/powerlens/actions/workflows/test.yml/badge.svg)](https://github.com/ssaserkar/powerlens/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/powerlens.svg)](https://pypi.org/project/powerlens/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**Measure the energy cost of AI inference on NVIDIA Jetson — no extra hardware needed.**

PowerLens reads the Jetson's built-in INA3221 power sensors via sysfs, correlates power measurements with individual AI inference events, and computes **joules per inference** — the metric that tells you how energy-efficient your model actually is.

**Validated on Jetson Orin Nano. Readings match tegrastats within 2%.**

---

## What Makes PowerLens Different

No other tool does this from the command line:

```bash
# Profile a TensorRT model's energy consumption
powerlens profile --onnx model.onnx --runs 50

# Compare two models side-by-side
powerlens compare model_a.onnx model_b.onnx

# Find the most energy-efficient power mode
sudo powerlens power-modes --onnx model.onnx

# Analyze energy scaling with iteration count
powerlens batch-scaling --onnx model.onnx --batches 1,10,50,100
```

---

## Real Results from Jetson Orin Nano

### Model Complexity vs Energy

Tested with custom conv nets (light/medium/heavy) on Jetson Orin Nano MAXN_SUPER mode:

```
Model    Latency   Energy/inf    Avg Power   Peak Power   GPU Util
----------------------------------------------------------------------
     light       1.3ms      0.011J       11.5W       11.8W       22%
    medium       7.2ms      0.130J       30.0W       31.6W       57%
     heavy      33.5ms      1.060J       31.8W       33.3W       54%

→ Heavy model uses 97.7x more energy per inference than light model
```

### Power Mode Comparison (ResNet18 FP16)

```
Power Mode Comparison
======================================================================
Mode               Latency   Energy/inf    Avg Power   Efficiency
----------------------------------------------------------------------
15W                   2.9ms      0.015J       11.6W       68.0 inf/J
25W                   2.9ms      0.015J       11.8W       69.1 inf/J
MAXN_SUPER            2.9ms      0.015J       12.1W       65.4 inf/J
----------------------------------------------------------------------
Most efficient: 25W (69.1 inf/J)

→ 25W mode is more energy efficient than MAXN_SUPER for this model
```

### Sustained GPU Stress (90 seconds)

```
Idle: 10.4W → Load: 37.0W (255% increase)
GPU: 60.8°C → 63.3°C (+2.5°C over 90s)
GPU utilization: avg 95%, freq 1020MHz sustained
Throttling: NO ✓
```

### TensorRT Profiling with Full Telemetry

```
powerlens profile --onnx resnet18.onnx --runs 20

PowerLens Inference Energy Report
==========================================
Inferences:         20
Sample rate:        99.8 Hz

Energy/inference:   0.6778 +/- 0.0768 J
  Min:              0.4738 J
  Max:              0.7714 J

Power (avg):        12.57 W
Power (peak):       12.88 W
Power (idle):       7.71 W

Rail breakdown (avg power):
  VDD_IN               7.73 W (64%)
  VDD_CPU_GPU_CV       2.48 W (21%)
  VDD_SOC              1.79 W (15%)

Thermal Analysis
==========================================
  gpu-thermal          avg=38.6°C  max=39.2°C
  cpu-thermal          avg=37.3°C  max=37.5°C
✓ No thermal throttling detected

GPU Utilization
==========================================
  GPU util:  avg=41%  max=94%  min=0%
```

---

## Quick Start

```bash
pip install powerlens
```

### On Jetson — Profile a TensorRT Model

```bash
pip install powerlens[jetson]
powerlens detect                          # Check available sensors
powerlens profile --onnx model.onnx       # Profile your model
powerlens compare a.onnx b.onnx           # Compare two models
```

### On Any Machine — Development with Mock Sensor

```bash
powerlens demo --runs 20 --output results/
```

### Python API — Profile Your Own Code

```python
import powerlens

# Simple: context manager
with powerlens.context() as ctx:
    for image in test_images:
        ctx.mark_inference_start()
        result = model.infer(image)
        ctx.mark_inference_end()

report = ctx.report()
print(report.summary())
```

```python
# Even simpler: one-call profiling
report = powerlens.profile(num_runs=50, real_workload=True)
print(report.summary())
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `powerlens demo` | Run demo with mock or real sensor |
| `powerlens demo --real` | Run with real sensor + CPU stress workload |
| `powerlens detect` | Detect available power sensors |
| `powerlens profile --onnx model.onnx` | Profile TensorRT model energy |
| `powerlens compare a.onnx b.onnx` | Compare energy efficiency of two models |
| `powerlens power-modes --onnx model.onnx` | Profile across Jetson power modes (needs sudo) |
| `powerlens batch-scaling --onnx model.onnx` | Analyze energy scaling with iteration count |

---

## How It Works

1. **Sensor Layer:** Reads INA3221 power monitor via sysfs with auto-detected rail names (VDD_IN, VDD_CPU_GPU_CV, VDD_SOC)
2. **Sampler:** Background thread captures power at 100Hz
3. **GPU Monitor:** Reads GPU utilization % and clock frequency from sysfs
4. **Thermal Monitor:** Reads 9 thermal zones, detects throttling events
5. **Profiler:** Wraps your inference code, recording start/end timestamps
6. **Analysis:** Aligns power samples with inference events, integrates power over time using trapezoidal rule to compute energy (joules)
7. **TensorRT Runner:** Automatically builds engines from ONNX, manages CUDA memory, handles dynamic iteration counts

```
Power (W)
38 ┤     ╭────────────────────────────╮
30 ┤   ╭─╯  GPU inference running     ╰─╮
20 ┤  ╭╯                                ╰╮
10 ┤──╯  idle                       idle  ╰──
   └──────────────────────────────────────── Time
        ↑ 10.4W                    37.0W ↑
```

---

## Features

### Power Measurement

- **Per-inference energy:** Joules per inference via trapezoidal integration
- **Per-rail breakdown:** VDD_IN, VDD_CPU_GPU_CV, VDD_SOC measured separately
- **Auto-detected rail names:** Reads sysfs label files, no hardcoding
- **Background sampling:** Non-blocking 100Hz power measurement
- **Idle baseline:** Automatic idle power measurement for accurate net energy
- **Validated:** Matches tegrastats within 2%

### GPU Monitoring

- **GPU utilization %:** Real-time from sysfs
- **GPU clock frequency:** Current MHz from devfreq
- **Correlated with power:** See how GPU load affects energy

### Thermal Analysis

- **9 thermal zones:** CPU, GPU, SoC, junction temperature
- **Throttle detection:** Alerts when temperature exceeds threshold
- **Correlated with energy:** Detects when throttling increases energy per inference

### TensorRT Integration

- **One-command profiling:** `powerlens profile --onnx model.onnx`
- **Auto engine building:** Builds TensorRT FP16 engines from ONNX
- **Auto iteration tuning:** Detects optimal iterations per run for energy resolution
- **Model comparison:** Side-by-side energy efficiency comparison
- **Power mode sweep:** Profiles across all Jetson nvpmodel modes
- **No pycuda dependency:** Uses ctypes for CUDA memory management

### Output

- **CSV export:** Per-inference summary and raw power samples
- **Power trace plots:** Publication-quality matplotlib visualizations
- **Comparison tables:** Model vs model, mode vs mode
- **Timeline data:** Temperature, power, GPU util over time

### Development

- **Mock sensor:** Full development and testing without Jetson hardware
- **39 tests:** Covering sensors, sampler, energy analysis, session API
- **CI:** GitHub Actions on Python 3.9, 3.10, 3.11
- **Cross-platform:** Develop on Windows/Mac, deploy on Jetson

---

## Requirements

- Python 3.9+
- NVIDIA Jetson (Orin Nano, AGX Orin) for real measurements
- TensorRT (included with JetPack) for model profiling
- Works on any platform with mock sensor for development

### Jetson Setup

```bash
pip install powerlens[jetson]
sudo usermod -aG i2c $USER  # then re-login
powerlens detect             # verify sensors
```

---

## How is this different from...

| Tool | What it does | What PowerLens adds |
|------|-------------|---------------------|
| **tegrastats/jtop** | System power at ~1Hz | Per-inference energy, 100Hz sampling, workload correlation, GPU util |
| **PowerSensor3** | Lab-grade measurement with custom hardware at 20kHz | No hardware needed, AI-native, pip-installable, model comparison |
| **powertool** | Raw INA reads via external I2C adapter | Jetson-native, AI workload awareness, TensorRT integration |
| **Nsight Systems** | GPU compute profiling | Hardware-level power + thermal + GPU util in one report |

---

## Related Work

- [Chakraborty et al. (2024)](https://doi.org/) — Profiling concurrent vision inference on Jetson (compute-level)
- [Li & Zheng (2022)](https://doi.org/) — Profiling Jetson GPU devices for autonomous machines
- [Van der Vlugt et al. (2024)](https://doi.org/) — PowerSensor3: high-accuracy external power measurement
- [powertool](https://github.com/) — INA226 power measurement for TI boards

PowerLens complements these by providing per-inference energy measurement with GPU utilization and thermal correlation using on-board sensors with zero additional hardware.

---

## Project Structure

```
powerlens/
├── src/powerlens/
│   ├── sensors/          # INA3221, sysfs, mock, auto-detection, GPU monitor
│   ├── profiler/         # Sampler, session API, TensorRT runner
│   ├── analysis/         # Energy computation, thermal, power modes, batch scaling
│   ├── export/           # CSV export
│   ├── visualization/    # Matplotlib power trace plots
│   └── cli.py            # 7 CLI commands
├── tests/                # 39 tests
├── examples/
│   ├── demo_mock.py          # Mock sensor demo
│   ├── quickstart.py         # Profile your own code
│   ├── demo_tensorrt.py      # TensorRT profiling
│   ├── create_demo_model.py  # Generate light/medium/heavy models
│   └── full_showcase.py      # Complete 3-part showcase
└── docs/
```

---

## Full Showcase

Generate demo models and run the complete analysis:

```bash
cd examples/
python create_demo_model.py    # Creates light/medium/heavy ONNX models
python full_showcase.py        # Runs all 3 parts (~3 minutes)
```

Produces:

- Model comparison CSV and power trace plots
- Iteration scaling analysis
- 90-second sustained thermal stress test with timeline data
- 15 output files (CSVs, PNGs, timeline data)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where help is needed:**

- Sensor support for other Jetson platforms (Xavier NX, AGX Orin, TX2)
- PyTorch inference hook for automatic profiling
- Real-time terminal dashboard (TUI)
- PDF report generation
- MLflow / Weights & Biases integration

---

## License

[Apache 2.0](LICENSE) — use it in your research, your startup, your thesis.

---

## Citation

If PowerLens helps your research, please cite:

```bibtex
@software{powerlens2025,
  title={PowerLens: Per-Inference Energy Profiling for NVIDIA Jetson},
  author={Saserkar, S.},
  year={2025},
  url={https://github.com/ssaserkar/powerlens}
}
```