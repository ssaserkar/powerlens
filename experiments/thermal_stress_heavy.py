#!/usr/bin/env python3
"""
Thermal stress with heavy demo model — should draw 30-40W.
"""

import time
import ctypes
import threading
import numpy as np
from pathlib import Path


def cpu_stress_worker(stop_event):
    while not stop_event.is_set():
        a = np.random.randn(500, 500).astype(np.float32)
        b = np.random.randn(500, 500).astype(np.float32)
        np.dot(a, b)


def main():
    import tensorrt as trt

    MODEL = "models/demo_heavy.onnx"
    DURATION_S = 300
    BATCH_SIZE = 1  # Start with 1 — this model is already heavy

    print(f"Building TensorRT engine for {MODEL}...")

    from powerlens.profiler.tensorrt_runner import (
        build_engine_from_onnx,
    )
    engine = build_engine_from_onnx(MODEL)
    context = engine.create_execution_context()

    # Print engine info
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        print(f"  {name}: {list(shape)}")

    # CUDA setup
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

    buffer_list = []
    device_ptrs = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(context.get_tensor_shape(name))
        # Replace dynamic dims with 1
        shape = [1 if s == -1 else s for s in shape]
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_data = np.random.randn(*shape).astype(dtype)

        ptr = ctypes.c_void_p()
        ret = cuda.cudaMalloc(
            ctypes.byref(ptr), host_data.nbytes
        )
        if ret != 0:
            print(f"cudaMalloc FAILED for {name} "
                  f"({host_data.nbytes} bytes)")
            return

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
    print("Warming up...")
    for _ in range(10):
        context.execute_v2(buffer_list)
    cuda.cudaDeviceSynchronize()

    # Quick latency check
    start_t = time.monotonic()
    for _ in range(10):
        context.execute_v2(buffer_list)
    cuda.cudaDeviceSynchronize()
    lat = (time.monotonic() - start_t) / 10 * 1000
    print(f"Single inference latency: {lat:.1f} ms")

    def read_temps():
        temps = {}
        for zone in sorted(
            Path("/sys/class/thermal").glob("thermal_zone*")
        ):
            try:
                temp = int(
                    (zone / "temp").read_text(
                        errors="replace"
                    ).strip()
                ) / 1000
                name = (zone / "type").read_text(
                    errors="replace"
                ).strip()
                temps[name] = temp
            except (IOError, ValueError, TypeError):
                pass
        return temps

    def read_power():
        try:
            hwmon = Path("/sys/bus/i2c/devices/1-0040/hwmon")
            hwmon_dir = next(hwmon.iterdir())
            powers = {}
            for n in range(1, 4):
                label = (
                    hwmon_dir / f"in{n}_label"
                ).read_text().strip()
                v = int(
                    (hwmon_dir / f"in{n}_input"
                     ).read_text().strip()
                ) / 1000
                i = int(
                    (hwmon_dir / f"curr{n}_input"
                     ).read_text().strip()
                ) / 1000
                powers[label] = round(v * i, 1)
            return powers
        except Exception:
            return {}

    def read_gpu_util():
        try:
            val = int(Path(
                "/sys/devices/platform/bus@0/"
                "17000000.gpu/load"
            ).read_text().strip())
            return val / 10
        except (IOError, ValueError):
            return 0.0

    # Start CPU stress
    print("Starting CPU stress on all cores...")
    stop_event = threading.Event()
    cpu_threads = []
    for _ in range(6):
        t = threading.Thread(
            target=cpu_stress_worker,
            args=(stop_event,),
            daemon=True,
        )
        t.start()
        cpu_threads.append(t)

    print(f"\nStarting {DURATION_S}s stress with HEAVY model "
          f"+ CPU stress...")
    print(f"{'Time':>6s} {'GPU°C':>7s} {'CPU°C':>7s} "
          f"{'TJ°C':>7s} {'VDD_IN':>7s} {'CPU_GPU':>8s} "
          f"{'SOC':>6s} {'GPU%':>5s} {'Infs':>8s}")
    print("-" * 72)

    start = time.monotonic()
    last_log = start
    inference_count = 0
    temp_log = []

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= DURATION_S:
            break

        context.execute_v2(buffer_list)
        inference_count += 1

        if time.monotonic() - last_log >= 5.0:
            cuda.cudaDeviceSynchronize()
            temps = read_temps()
            powers = read_power()
            gpu_util = read_gpu_util()

            gpu_t = temps.get("gpu-thermal", 0)
            cpu_t = temps.get("cpu-thermal", 0)
            tj_t = temps.get("tj-thermal", 0)
            vdd_in = powers.get("VDD_IN", 0)
            cpu_gpu = powers.get("VDD_CPU_GPU_CV", 0)
            soc = powers.get("VDD_SOC", 0)

            entry = {
                "time_s": round(elapsed, 1),
                "gpu_c": gpu_t,
                "cpu_c": cpu_t,
                "tj_c": tj_t,
                "vdd_in_w": vdd_in,
                "cpu_gpu_w": cpu_gpu,
                "soc_w": soc,
                "gpu_util_pct": gpu_util,
                "total_inferences": inference_count,
            }
            temp_log.append(entry)

            print(
                f"{elapsed:>5.0f}s {gpu_t:>6.1f} "
                f"{cpu_t:>6.1f} {tj_t:>6.1f} "
                f"{vdd_in:>6.1f}W {cpu_gpu:>7.1f}W "
                f"{soc:>5.1f}W {gpu_util:>4.0f}% "
                f"{inference_count:>7d}"
            )

            last_log = time.monotonic()

    cuda.cudaDeviceSynchronize()
    stop_event.set()
    for t in cpu_threads:
        t.join(timeout=2)

    final_temps = read_temps()
    total_time = time.monotonic() - start

    print(f"\n{'='*72}")
    print(f"Duration: {total_time:.0f}s")
    print(f"Model: {MODEL}")
    print(f"Total inferences: {inference_count}")
    print(f"Throughput: {inference_count/total_time:.1f} inf/s")
    print(f"Final GPU temp: "
          f"{final_temps.get('gpu-thermal', 0):.1f}°C")
    print(f"Final TJ temp: "
          f"{final_temps.get('tj-thermal', 0):.1f}°C")
    print(f"Throttle threshold: 85°C")
    print(f"Throttled: "
          f"{'YES!' if final_temps.get('tj-thermal', 0) >= 85 else 'No'}")

    import json
    results = {
        "duration_s": round(total_time, 1),
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "cpu_stress_threads": 6,
        "total_inferences": inference_count,
        "throughput_inf_s": round(
            inference_count / total_time, 1
        ),
        "temp_timeline": temp_log,
        "throttled": (
            final_temps.get("tj-thermal", 0) >= 85
        ),
    }

    out = Path(
        "experiments/results/thermal_stress_heavy.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out}")

    for ptr in device_ptrs:
        cuda.cudaFree(ptr)


if __name__ == "__main__":
    main()