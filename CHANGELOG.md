# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2025-03-17

### Added

- **Core Data Model**
  - `PowerSample` dataclass for standardized sensor readings
  - `InferenceEvent` dataclass for tracking inference start/end timestamps

- **Sensors**
  - `MockSensor` for development and testing without Jetson hardware
  - Sensor interface (`BaseSensor`) for future hardware driver implementations
  - INA3221 I2C sensor driver architecture

- **Profiler**
  - `PowerSampler` background thread for continuous power sampling at up to 100Hz
  - `PowerLensContext` context manager for profiling custom inference code
  - `powerlens.profile()` one-call profiling function
  - Inference start/end marking via `mark_inference_start()` / `mark_inference_end()`

- **Analysis**
  - Energy computation engine with trapezoidal integration
  - Per-inference energy (joules/inference) calculation
  - Per-rail power breakdown (GPU, CPU, system)
  - Idle baseline measurement for accurate net energy computation
  - Summary statistics (mean, std, min, max energy per inference)

- **Export**
  - CSV export for per-inference summary
  - CSV export for raw power samples

- **Visualization**
  - Matplotlib power trace plotting
  - Inference event overlay on power traces
  - Publication-quality figure output

- **CLI**
  - `powerlens demo` command for quick profiling
  - `--runs` flag to configure number of inference runs
  - `--output` flag to specify output directory

- **CI/CD**
  - GitHub Actions workflow testing Python 3.9, 3.10, 3.11
  - Ruff linting in CI pipeline
  - Coverage reporting

- **Testing**
  - 24 tests covering all modules
  - Full test coverage for mock sensor, sampler, profiler, analysis, export, and CLI

- **Documentation**
  - Professional README with usage examples, architecture, and related work
  - Citation block for academic use
  - CONTRIBUTING.md with development guidelines
  - This changelog

---

## [Unreleased]

### Planned

- Sysfs sensor driver for Jetson Orin Nano / AGX Orin
- I2C sensor driver for direct INA3221 register reads
- TensorRT inference hook for automatic profiling
- PyTorch integration
- Sensor configs for Xavier, TX2 platforms
- JSON export format
- Interactive HTML power trace visualization
- PyPI package publishing