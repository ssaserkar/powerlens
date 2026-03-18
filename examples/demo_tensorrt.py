"""
PowerLens Real Workload Demo — TensorRT on Jetson

Runs actual GPU inference using TensorRT and measures
energy per inference with real hardware sensors.

Usage:
    python examples/demo_tensorrt.py
    python examples/demo_tensorrt.py --onnx /path/to/model.onnx --runs 50
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def build_engine_from_onnx(onnx_path):
    """Build a TensorRT engine from an ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Loading ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

    # Enable FP16 if available
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled")

    print("  Building TensorRT engine (this may take a minute)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    print(f"  Engine built successfully")
    return engine


def load_engine(engine_path):
    """Load a pre-built TensorRT engine."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    print(f"Loading TensorRT engine: {engine_path}")
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"  Engine loaded successfully")
    return engine

def run_inference(engine, num_runs, warmup=5):
    """Run inference using TensorRT without pycuda."""
    import tensorrt as trt

    context = engine.create_execution_context()

    # Use numpy for host memory, tensorrt for device memory
    import numpy as np

    host_inputs = []
    host_outputs = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            host_data = np.random.randn(*shape).astype(dtype)
            host_inputs.append({"name": name, "data": host_data})
        else:
            host_data = np.zeros(shape, dtype=dtype)
            host_outputs.append({"name": name, "data": host_data})

    # Try using cuda-python (available on JetPack)
    try:
        from cuda import cudart

        # Allocate device memory
        device_buffers = {}
        for inp in host_inputs:
            nbytes = inp["data"].nbytes
            err, ptr = cudart.cudaMalloc(nbytes)
            device_buffers[inp["name"]] = ptr
            cudart.cudaMemcpy(ptr, inp["data"].ctypes.data, nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            context.set_tensor_address(inp["name"], ptr)

        for out in host_outputs:
            nbytes = out["data"].nbytes
            err, ptr = cudart.cudaMalloc(nbytes)
            device_buffers[out["name"]] = ptr
            context.set_tensor_address(out["name"], ptr)

        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            for _ in range(100):
                context.execute_v2(buffer_list)

        # Timed runs
        print(f"  Running {num_runs} inferences...")
        timestamps = []
        for i in range(num_runs):
            start = time.monotonic()
            # Run multiple iterations per "inference" to accumulate measurable energy
            for _ in range(100):
                context.execute_v2(buffer_list)
            end = time.monotonic()
            timestamps.append((start, end))
            time.sleep(0.05)  # 50ms gap for power to settle

        # Cleanup
        for ptr in device_buffers.values():
            cudart.cudaFree(ptr)

        return timestamps

    except ImportError:
        pass

    # Fallback: try using ctypes to call CUDA directly
    try:
        import ctypes

        cuda_lib = ctypes.CDLL("libcudart.so")

        # cudaMalloc
        cuda_malloc = cuda_lib.cudaMalloc
        cuda_malloc.restype = ctypes.c_int
        cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

        # cudaMemcpy
        cuda_memcpy = cuda_lib.cudaMemcpy
        cuda_memcpy.restype = ctypes.c_int
        cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_int]

        # cudaFree
        cuda_free = cuda_lib.cudaFree
        cuda_free.restype = ctypes.c_int
        cuda_free.argtypes = [ctypes.c_void_p]

        MEMCPY_H2D = 1
        MEMCPY_D2H = 2

        device_buffers = {}
        buffer_list = []

        for inp in host_inputs:
            nbytes = inp["data"].nbytes
            ptr = ctypes.c_void_p()
            cuda_malloc(ctypes.byref(ptr), nbytes)
            cuda_memcpy(ptr, inp["data"].ctypes.data, nbytes, MEMCPY_H2D)
            device_buffers[inp["name"]] = ptr.value
            context.set_tensor_address(inp["name"], ptr.value)
            buffer_list.append(ptr.value)

        for out in host_outputs:
            nbytes = out["data"].nbytes
            ptr = ctypes.c_void_p()
            cuda_malloc(ctypes.byref(ptr), nbytes)
            device_buffers[out["name"]] = ptr.value
            context.set_tensor_address(out["name"], ptr.value)
            buffer_list.append(ptr.value)

        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            for _ in range(100):
                context.execute_v2(buffer_list)

        # Timed runs
        print(f"  Running {num_runs} inferences...")
        timestamps = []
        for i in range(num_runs):
            start = time.monotonic()
            # Run multiple iterations per "inference" to accumulate measurable energy
            for _ in range(100):
                context.execute_v2(buffer_list)
            end = time.monotonic()
            timestamps.append((start, end))
            time.sleep(0.05)  # 50ms gap for power to settle

        # Cleanup
        for ptr_val in device_buffers.values():
            cuda_free(ctypes.c_void_p(ptr_val))

        return timestamps

    except Exception as e:
        print(f"ERROR: Could not allocate CUDA memory: {e}")
        print("Falling back to CPU-timed simulation")
        timestamps = []
        for i in range(num_runs):
            start = time.monotonic()
            time.sleep(0.005)  # Simulate inference time
            end = time.monotonic()
            timestamps.append((start, end))
            time.sleep(0.05)
        return timestamps
    

def main():
    parser = argparse.ArgumentParser(
        description="PowerLens TensorRT inference energy profiling"
    )
    parser.add_argument(
        "--onnx", type=str, default=None,
        help="Path to ONNX model (will build TensorRT engine)"
    )
    parser.add_argument(
        "--engine", type=str, default=None,
        help="Path to pre-built TensorRT engine"
    )
    parser.add_argument(
        "--runs", type=int, default=100,
        help="Number of inference runs (default: 100)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results_trt",
        help="Output directory (default: results_trt)"
    )
    args = parser.parse_args()

    # Default model if none specified
    if args.onnx is None and args.engine is None:
        default_onnx = os.path.expanduser(
            "~/containers/jetson-containers/data/models/onnx/"
            "cat_dog_epoch_100/resnet18.onnx"
        )
        if os.path.exists(default_onnx):
            args.onnx = default_onnx
            print(f"Using default model: {default_onnx}")
        else:
            print("ERROR: No model specified and default not found.")
            print("Usage: python examples/demo_tensorrt.py --onnx /path/to/model.onnx")
            sys.exit(1)

    # Load or build engine
    import tensorrt as trt
    if args.engine:
        engine = load_engine(args.engine)
    else:
        engine = build_engine_from_onnx(args.onnx)

    # Print model info
    print(f"\nModel info:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        io = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"  {io}: {name} {list(shape)}")

    # Setup PowerLens
    from powerlens.profiler.session import PowerLensContext
    from powerlens.sensors.auto import detect_sensor
    from powerlens.export.csv_export import export_summary_csv, export_raw_csv
    from powerlens.visualization.plots import plot_power_trace

    sensor = detect_sensor(use_mock_fallback=False)
    print(f"\nPower sensor: {type(sensor).__name__}")

    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=100)

    print(f"\nProfiling {args.runs} inferences...")
    with ctx:
        timestamps = run_inference(engine, args.runs)
        for start, end in timestamps:
            ctx._inference_timestamps.append((start, end))

    report = ctx.report()
    samples = ctx._sampler.get_samples()

    # Print report
    print(report.summary())

    # Print per-inference details for first 5
    print("First 5 inferences:")
    for inf in report.inferences[:5]:
        print(
            f"  #{inf.index}: {inf.energy_j:.4f} J, "
            f"{inf.duration_s*1000:.1f} ms, "
            f"{inf.avg_power_w:.2f} W avg"
        )
    print()

    # Export
    os.makedirs(args.output, exist_ok=True)

    csv_path = export_summary_csv(
        report, os.path.join(args.output, "energy_summary.csv")
    )
    print(f"Summary CSV: {csv_path}")

    raw_path = export_raw_csv(
        samples, os.path.join(args.output, "raw_samples.csv")
    )
    print(f"Raw CSV:     {raw_path}")

    model_name = os.path.basename(args.onnx or args.engine or "model")
    plot_path = plot_power_trace(
        samples=samples,
        report=report,
        filepath=os.path.join(args.output, "power_trace.png"),
        title=f"PowerLens — {model_name} ({args.runs} inferences)",
    )
    print(f"Plot:        {plot_path}")


if __name__ == "__main__":
    main()