#!/usr/bin/env python3
"""
FP16 vs FP32 comparison for all 5 models.
Runs at MAXN mode for fair comparison.
"""

import time
import ctypes
import numpy as np
import tensorrt as trt
from pathlib import Path


def build_engine(onnx_path, use_fp16=True):
    """Build TRT engine with or without FP16."""
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [
                str(parser.get_error(i))
                for i in range(parser.num_errors)
            ]
            raise RuntimeError(f"Parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 1 << 28
    )

    # Handle dynamic shapes
    has_dynamic = False
    for i in range(network.num_inputs):
        if -1 in network.get_input(i).shape:
            has_dynamic = True
            break

    if has_dynamic:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = list(inp.shape)
            static = [1 if s == -1 else s for s in shape]
            profile.set_shape(
                inp.name,
                min=static,
                opt=static,
                max=static,
            )
        config.add_optimization_profile(profile)

    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(
        network, config
    )
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(trt_logger)
    return runtime.deserialize_cuda_engine(engine_bytes)


def measure(engine, num_runs=100, warmup=50):
    """Run inference and measure latency + power."""
    cuda = ctypes.CDLL("libcudart.so")
    cuda.cudaMalloc.restype = ctypes.c_int
    cuda.cudaMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
    ]
    cuda.cudaMemcpy.restype = ctypes.c_int
    cuda.cudaMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int
    ]
    cuda.cudaDeviceSynchronize.restype = ctypes.c_int
    cuda.cudaFree.restype = ctypes.c_int
    cuda.cudaFree.argtypes = [ctypes.c_void_p]

    context = engine.create_execution_context()
    buffer_list = []
    device_ptrs = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(context.get_tensor_shape(name))
        shape = [1 if s == -1 else s for s in shape]
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_data = np.random.randn(*shape).astype(dtype)

        ptr = ctypes.c_void_p()
        cuda.cudaMalloc(ctypes.byref(ptr), host_data.nbytes)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cuda.cudaMemcpy(
                ptr, host_data.ctypes.data,
                host_data.nbytes, 1
            )
        context.set_tensor_address(name, ptr.value)
        buffer_list.append(ptr.value)
        device_ptrs.append(ptr)

    cuda.cudaDeviceSynchronize()

    # Warmup
    for _ in range(warmup):
        context.execute_v2(buffer_list)
    cuda.cudaDeviceSynchronize()

    # Read power during timed runs
    def read_power():
        try:
            hwmon = Path("/sys/bus/i2c/devices/1-0040/hwmon")
            hwmon_dir = next(hwmon.iterdir())
            for n in range(1, 4):
                label = (
                    hwmon_dir / f"in{n}_label"
                ).read_text().strip()
                if label == "VDD_IN":
                    v = int(
                        (hwmon_dir / f"in{n}_input"
                         ).read_text().strip()
                    ) / 1000
                    i = int(
                        (hwmon_dir / f"curr{n}_input"
                         ).read_text().strip()
                    ) / 1000
                    return v * i
        except Exception:
            pass
        return 0.0

    # Timed runs
    power_samples = []
    latencies = []

    for _ in range(num_runs):
        cuda.cudaDeviceSynchronize()
        p = read_power()
        power_samples.append(p)

        start = time.monotonic()
        context.execute_v2(buffer_list)
        cuda.cudaDeviceSynchronize()
        end = time.monotonic()

        latencies.append((end - start) * 1000)

    # Cleanup
    for ptr in device_ptrs:
        cuda.cudaFree(ptr)
    del context

    avg_latency = sum(latencies) / len(latencies)
    avg_power = sum(power_samples) / len(power_samples)
    energy_per_inf = avg_power * avg_latency / 1000  # W * s = J

    return {
        "latency_ms": round(avg_latency, 2),
        "avg_power_w": round(avg_power, 1),
        "energy_per_inf_mj": round(energy_per_inf * 1000, 2),
        "inferences_per_joule": round(
            1 / energy_per_inf if energy_per_inf > 0 else 0, 1
        ),
    }


def main():
    models = {
        "MobileNetV2": "models/mobilenetv2.onnx",
        "ResNet-18": "models/resnet18.onnx",
        "ResNet-34": "models/resnet34.onnx",
        "ResNet-50": "models/resnet50.onnx",
        "EfficientNet-B0": "models/efficientnet_b0.onnx",
    }

    print("FP16 vs FP32 Comparison (MAXN_SUPER mode)")
    print("=" * 75)
    print()

    results = {}

    for name, path in models.items():
        if not Path(path).exists():
            print(f"  SKIP: {path} not found")
            continue

        print(f"  {name}:")

        # FP16
        print(f"    Building FP16 engine...", end="", flush=True)
        engine_fp16 = build_engine(path, use_fp16=True)
        print(" measuring...", end="", flush=True)
        r16 = measure(engine_fp16)
        del engine_fp16
        print(f" done")

        # FP32
        print(f"    Building FP32 engine...", end="", flush=True)
        engine_fp32 = build_engine(path, use_fp16=False)
        print(" measuring...", end="", flush=True)
        r32 = measure(engine_fp32)
        del engine_fp32
        print(f" done")

        results[name] = {"fp16": r16, "fp32": r32}

        speedup = r32["latency_ms"] / r16["latency_ms"]
        energy_ratio = r32["energy_per_inf_mj"] / r16["energy_per_inf_mj"]

        print(f"    FP16: {r16['latency_ms']:>6.2f}ms  "
              f"{r16['energy_per_inf_mj']:>6.2f}mJ  "
              f"{r16['avg_power_w']:>5.1f}W  "
              f"{r16['inferences_per_joule']:>6.1f} inf/J")
        print(f"    FP32: {r32['latency_ms']:>6.2f}ms  "
              f"{r32['energy_per_inf_mj']:>6.2f}mJ  "
              f"{r32['avg_power_w']:>5.1f}W  "
              f"{r32['inferences_per_joule']:>6.1f} inf/J")
        print(f"    → FP16 is {speedup:.1f}× faster, "
              f"{energy_ratio:.1f}× more energy efficient")
        print()

    # Summary table
    print()
    print("Summary Table")
    print("=" * 75)
    print(f"{'Model':20s} {'FP16 ms':>8s} {'FP32 ms':>8s} "
          f"{'Speedup':>8s} {'FP16 mJ':>8s} {'FP32 mJ':>8s} "
          f"{'E ratio':>8s}")
    print("-" * 75)

    for name, r in results.items():
        speedup = r["fp32"]["latency_ms"] / r["fp16"]["latency_ms"]
        e_ratio = (
            r["fp32"]["energy_per_inf_mj"]
            / r["fp16"]["energy_per_inf_mj"]
        )
        print(f"{name:20s} "
              f"{r['fp16']['latency_ms']:>8.2f} "
              f"{r['fp32']['latency_ms']:>8.2f} "
              f"{speedup:>7.1f}× "
              f"{r['fp16']['energy_per_inf_mj']:>8.2f} "
              f"{r['fp32']['energy_per_inf_mj']:>8.2f} "
              f"{e_ratio:>7.1f}×")

    # Save
    import json
    out = Path("experiments/results/fp16_vs_fp32.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()