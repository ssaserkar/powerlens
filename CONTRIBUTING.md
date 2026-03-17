# Contributing to PowerLens

Thank you for your interest in contributing to PowerLens! This project aims to make energy profiling accessible for edge AI developers.

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/powerlens.git
cd powerlens
```

### 2. Set up development environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate.ps1  # Windows

pip install -e ".[dev]"
```

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Run linter

```bash
ruff check src/ tests/ --select E,F,W --ignore E501
```

---

## How to Contribute

### Bug Reports

- Open a [GitHub Issue](https://github.com/ssaserkar/powerlens/issues)
- Include: Python version, OS, Jetson model (if applicable), and steps to reproduce

### Code Contributions

1. Create a branch: `git checkout -b feature/your-feature`
2. Write tests for your changes
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure linter passes: `ruff check src/ tests/ --select E,F,W --ignore E501`
5. Submit a Pull Request

---

## Areas Where Help Is Needed

- **Jetson platform support:** Sensor configs for Xavier, TX2, AGX Orin
- **TensorRT integration:** Automatic inference tracking hooks
- **PyTorch integration:** Energy profiling for PyTorch inference
- **Documentation:** Tutorials, Jetson setup guides
- **Benchmarks:** Energy profiles of popular models (YOLO, ResNet, MobileNet)

---

## Adding Support for a New Jetson Platform

1. Identify the INA3221 I2C address and bus number on your board
2. Identify the power rail names and shunt resistor values
3. Add a config in `src/powerlens/sensors/jetson.py`
4. Test with `powerlens demo` on the actual hardware
5. Submit a PR with your config and test results

---

## Code Style

- Python 3.9+ compatible
- Type hints on all public functions
- Docstrings on all public classes and functions
- Tests for all new functionality

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).