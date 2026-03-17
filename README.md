# ⚡ PowerLens

[![Tests](https://github.com/ssaserkar/powerlens/actions/workflows/test.yml/badge.svg)](https://github.com/ssaserkar/powerlens/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**Measure the energy cost of AI inference on NVIDIA Jetson — no extra hardware needed.**

PowerLens reads the Jetson's built-in INA3221 power sensors directly via I2C and correlates power measurements with individual AI inference events to compute **joules per inference** — the metric that tells you how energy-efficient your model actually is.

---

## Why?

You can measure how *fast* your model runs (latency) and how *accurate* it is (mAP). But how much *energy* does each inference cost?

On battery-powered robots, drones, and edge devices, energy per inference determines how long your device runs before it dies. On large-scale deployments of thousands of Jetson devices, it determines your electricity bill.

Existing tools (tegrastats, jtop) report power at ~1Hz with no connection to your AI workload. PowerLens samples at up to 100Hz and automatically computes the energy cost of each inference.

---

## Quick Start

```bash
pip install powerlens
```

### One-call profiling

```python
import powerlens

report = powerlens.profile(num_runs=100, load_level=0.8)
print(report.summary())
```

### Profile your own inference code

```python
from powerlens.profiler.session import PowerLensContext

ctx = PowerLensContext(sample_rate_hz=100)

with ctx:
    for image in test_images:
        ctx.mark_inference_start()
        result = model.infer(image)
        ctx.mark_inference_end()

report = ctx.report()
print(report.summary())
```

### Command line

```bash
powerlens demo --runs 20 --output results/
```

---

## Example Output

```
PowerLens Inference Energy Report
==========================================
Inferences:         20
Sample rate:        98.7 Hz

Energy/inference:   0.4581 +/- 0.0549 J
  Min:              0.4062 J
  Max:              0.6071 J

Power (avg):        9.02 W
Power (peak):       10.26 W
Power (idle):       3.19 W

Total energy:       4.5807 J
Total duration:     0.48 s

Efficiency:         2.2 inferences/J

Rail breakdown (avg power):
  VDD_GPU_SOC          5.59 W (62%)
  VDD_CPU_CV           2.25 W (25%)
  VIN_SYS_5V0          1.21 W (13%)
```

---

## How It Works

1. **Sensor Layer:** Reads INA3221 power monitor IC via I2C or sysfs
2. **Sampler:** Background thread captures power at configurable rate (up to 100Hz)
3. **Profiler:** Wraps your inference code, recording start/end timestamps
4. **Analysis:** Aligns power samples with inference events, integrates power over time using trapezoidal rule to compute energy (joules)

```
Power (W)
12 ┤        ╭──╮     ╭──╮     ╭──╮
10 ┤      ╭─╯  ╰─╮ ╭─╯  ╰─╮ ╭─╯  ╰─╮
 8 ┤    ╭─╯      ╰─╯      ╰─╯      ╰─╮
 6 ┤  ╭─╯                              ╰─╮
 4 ┤──╯  ↑inf1    ↑inf2    ↑inf3        ╰──
 2 ┤     0.47J    0.45J    0.51J
   └──────────────────────────────────────── Time
```

---

## Features

- **Per-inference energy:** Joules per inference via trapezoidal integration
- **Per-rail breakdown:** GPU, CPU, and system power measured separately
- **Background sampling:** Non-blocking power measurement at up to 100Hz
- **Idle baseline:** Automatic idle power measurement for accurate net energy
- **CSV export:** Per-inference summary and raw power samples
- **Power trace plots:** Publication-quality matplotlib visualizations
- **Mock sensor:** Full development and testing without Jetson hardware
- **CLI:** Command-line interface for quick profiling

---

## Requirements

- Python 3.9+
- NVIDIA Jetson (Orin Nano, AGX Orin) for real measurements
- Works on any platform with mock sensor for development

### Jetson-specific

```bash
pip install powerlens[jetson]
sudo usermod -aG i2c $USER  # then re-login
```

---

## How is this different from...

| Tool | What it does | What PowerLens adds |
|------|-------------|---------------------|
| **tegrastats/jtop** | System power at ~1Hz | Per-inference energy, 100Hz sampling, workload correlation |
| **PowerSensor3** | Lab-grade measurement with custom hardware | No hardware needed, AI-native, pip-installable |
| **powertool** | Raw INA reads via external I2C adapter | Jetson-native, AI workload awareness, modern Python |
| **Nsight Systems** | GPU compute profiling | Hardware-level power measurement |

---

## Related Work

- [Chakraborty et al. (2024)](https://doi.org/) — Profiling concurrent vision inference on Jetson (compute-level)
- [Li & Zheng (2022)](https://doi.org/) — Profiling Jetson GPU devices for autonomous machines
- [Van der Vlugt et al. (2024)](https://doi.org/) — PowerSensor3: high-accuracy external power measurement
- [powertool](https://github.com/) — INA226 power measurement for TI boards

PowerLens complements these by providing per-inference energy measurement using on-board sensors with zero additional hardware.

---

## Project Structure

```
powerlens/
├── src/powerlens/
│   ├── sensors/        # Hardware sensor drivers (INA3221, sysfs, mock)
│   ├── profiler/       # Sampling engine and session management
│   ├── analysis/       # Energy computation (trapezoidal integration)
│   ├── export/         # CSV export
│   ├── visualization/  # Matplotlib plotting
│   └── cli.py          # Command-line interface
├── tests/              # 24 tests covering all modules
└── examples/           # Runnable demo scripts
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where help is needed:**

- Sensor support for other Jetson platforms (Xavier, TX2)
- TensorRT inference hook for automatic profiling
- PyTorch integration
- Additional visualization options

---

## License

[Apache 2.0](LICENSE) — use it in your research, your startup, your thesis.

---

## Citation

If PowerLens helps your research, please cite:

```bibtex
@software{powerlens2025,
  title={PowerLens: Per-Inference Energy Profiling for NVIDIA Jetson},
  author={Aserkar, S.},
  year={2025},
  url={https://github.com/ssaserkar/powerlens}
}
```