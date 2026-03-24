#!/usr/bin/env python3
"""
Thermal stress test — sustained GPU inference for 5 minutes.
Logs temperature every second.
"""

import time
import ctypes
import numpy as np
from pathlib import Path
from datetime import datetime

# Build TRT engine once, then run inference in tight loop
def main():
    import tensorrt as trt

    MODEL = "models/resnet50.onnx"
    DURATION_S = 300  # 5 minutes
    
    print("Building TensorRT engine...")
    
    # Use PowerLens's engine builder (handles dynamic shapes)
    from powerlens.profiler.tensorrt_runner import build_engine_from_onnx
    engine = build_engine_from_onnx(MODEL)
    context = engine.create_execution_context()
    
    # Allocate buffers
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
        shape = engine.get_tensor_shape(name)
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
    print("Warming up...")
    for _ in range(50):
        context.execute_v2(buffer_list)
    cuda.cudaDeviceSynchronize()
    
    # Read temps function
    def read_temps():
        temps = {}
        thermal_base = Path("/sys/class/thermal")
        for zone in sorted(thermal_base.glob("thermal_zone*")):
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
        """Read VDD_IN power from sysfs."""
        try:
            hwmon = Path(
                "/sys/bus/i2c/devices/1-0040/hwmon"
            )
            hwmon_dir = next(hwmon.iterdir())
            # Find VDD_IN channel
            for n in range(1, 4):
                label = (
                    hwmon_dir / f"in{n}_label"
                ).read_text().strip()
                if label == "VDD_IN":
                    v = int(
                        (hwmon_dir / f"in{n}_input"
                         ).read_text().strip()
                    ) / 1000  # mV to V
                    i = int(
                        (hwmon_dir / f"curr{n}_input"
                         ).read_text().strip()
                    ) / 1000  # mA to A
                    return v * i
        except Exception:
            pass
        return 0.0
    
    # Main stress loop
    print(f"\nStarting {DURATION_S}s thermal stress test...")
    print(f"{'Time':>6s} {'GPU°C':>7s} {'CPU°C':>7s} "
          f"{'TJ°C':>7s} {'Power':>7s} {'Infs':>8s}")
    print("-" * 50)
    
    start = time.monotonic()
    last_log = start
    inference_count = 0
    temp_log = []
    
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= DURATION_S:
            break
        
        # Run inference (no sync between — maximum throughput)
        context.execute_v2(buffer_list)
        inference_count += 1
        
        # Log every 5 seconds
        if time.monotonic() - last_log >= 5.0:
            cuda.cudaDeviceSynchronize()
            temps = read_temps()
            power = read_power()
            
            gpu_t = temps.get("gpu-thermal", 0)
            cpu_t = temps.get("cpu-thermal", 0)
            tj_t = temps.get("tj-thermal", 0)
            
            entry = {
                "time_s": round(elapsed, 1),
                "gpu_c": gpu_t,
                "cpu_c": cpu_t,
                "tj_c": tj_t,
                "power_w": round(power, 1),
                "total_inferences": inference_count,
            }
            temp_log.append(entry)
            
            print(f"{elapsed:>5.0f}s {gpu_t:>6.1f} {cpu_t:>6.1f} "
                  f"{tj_t:>6.1f} {power:>6.1f}W "
                  f"{inference_count:>7d}")
            
            last_log = time.monotonic()
    
    cuda.cudaDeviceSynchronize()
    
    # Final temps
    final_temps = read_temps()
    total_time = time.monotonic() - start
    
    print(f"\n{'='*50}")
    print(f"Duration: {total_time:.0f}s")
    print(f"Total inferences: {inference_count}")
    print(f"Throughput: {inference_count/total_time:.1f} inf/s")
    print(f"Final GPU temp: {final_temps.get('gpu-thermal', 0):.1f}°C")
    print(f"Final TJ temp: {final_temps.get('tj-thermal', 0):.1f}°C")
    print(f"Throttle threshold: 85°C")
    
    # Save results
    import json
    results = {
        "duration_s": round(total_time, 1),
        "total_inferences": inference_count,
        "throughput_inf_s": round(inference_count / total_time, 1),
        "initial_temps": temp_log[0] if temp_log else {},
        "final_temps": {
            "gpu_c": final_temps.get("gpu-thermal", 0),
            "cpu_c": final_temps.get("cpu-thermal", 0),
            "tj_c": final_temps.get("tj-thermal", 0),
        },
        "temp_timeline": temp_log,
        "throttled": final_temps.get("tj-thermal", 0) >= 85,
    }
    
    out = Path("experiments/results/thermal_stress_sustained.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")
    
    # Cleanup
    for ptr in device_ptrs:
        cuda.cudaFree(ptr)


if __name__ == "__main__":
    main()