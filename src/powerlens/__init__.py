"""
PowerLens: Per-inference energy profiling for NVIDIA Jetson.

Quick start:
    import powerlens

    # Simple profiling with auto-detected sensor
    report = powerlens.profile(num_runs=50)
    print(report.summary())

    # Profile your own inference code
    with powerlens.context() as ctx:
        for image in images:
            ctx.mark_inference_start()
            result = model.infer(image)
            ctx.mark_inference_end()
    report = ctx.report()
    print(report.summary())
"""

__version__ = "0.1.1"

from powerlens.profiler.session import profile, PowerLensContext  # noqa: F401


def context(sample_rate_hz: float = 100.0, sensor=None):
    """Create a profiling context with auto-detected sensor.

    Usage:
        import powerlens

        with powerlens.context() as ctx:
            ctx.mark_inference_start()
            result = model.infer(image)
            ctx.mark_inference_end()

        report = ctx.report()
        print(report.summary())
    """
    if sensor is None:
        from powerlens.sensors.auto import detect_sensor
        sensor = detect_sensor(use_mock_fallback=True)
    return PowerLensContext(sensor=sensor, sample_rate_hz=sample_rate_hz)
