# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.3   | ✅ Current release |

## Reporting a Vulnerability

If you discover a security vulnerability in PowerLens, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **soham339@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You can expect:
- Acknowledgment within 48 hours
- An assessment within 1 week
- A fix or mitigation plan within 2 weeks

## Scope

PowerLens reads hardware sensors via sysfs (read-only). It does not:
- Run with elevated privileges (except `power-modes` which requires sudo for nvpmodel)
- Open network connections
- Execute arbitrary code from user input
- Store or transmit sensitive data

The primary security considerations are:
- **File path injection**: PowerLens reads sysfs paths. Malicious sysfs paths could theoretically be crafted on a compromised system.
- **ONNX model loading**: The `profile` command loads ONNX models via TensorRT. Only load models from trusted sources.
- **sudo usage**: The `power-modes` command requires sudo to change nvpmodel settings. This is the only elevated operation.

## Dependencies

PowerLens depends on:
- numpy
- matplotlib
- smbus2 (optional, for direct I2C)
- TensorRT (optional, for model profiling)

We monitor dependencies for known vulnerabilities and update as needed.