"""
TensorRT inference runner for automated energy profiling.

Loads a TensorRT engine or ONNX model, runs inference,
and returns timestamps for energy correlation.

No pycuda dependency — uses ctypes for CUDA memory management.
"""

import time
import logging
import ctypes
import numpy as np

logger = logging.getLogger(__name__)

MEMCPY_H2D = 1
MEMCPY_D2H = 2


def _get_cuda_lib():
    """Load libcudart.so for memory management."""
    try:
        return ctypes.CDLL("libcudart.so")
    except OSError:
        raise RuntimeError(
            "libcudart.so not found. "
            "Are you running on a Jetson with CUDA installed?"
        )


def build_engine_from_onnx(onnx_path: str):
    """Build a TensorRT engine from an ONNX model."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Loading ONNX model: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"Failed to parse ONNX: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")

    logger.info("Building TensorRT engine...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine")

    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    return engine


def load_engine(engine_path: str):
    """Load a pre-built TensorRT engine."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)

    logger.info("Loading TensorRT engine: %s", engine_path)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def get_engine_info(engine) -> dict:
    """Get input/output tensor information from engine."""
    import tensorrt as trt

    info = {"inputs": [], "outputs": []}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))
        mode = engine.get_tensor_mode(name)
        dtype = str(engine.get_tensor_dtype(name))

        tensor_info = {"name": name, "shape": shape, "dtype": dtype}
        if mode == trt.TensorIOMode.INPUT:
            info["inputs"].append(tensor_info)
        else:
            info["outputs"].append(tensor_info)
    return info


def run_trt_inference(engine, num_runs: int, warmup: int = 5,
                      iterations_per_run: int = 1):
    """Run TensorRT inference and return timestamps.

    Args:
        engine: TensorRT engine.
        num_runs: Number of profiling runs.
        warmup: Number of warmup runs.
        iterations_per_run: Model executions per profiling run.
                           Increase for very fast models (<5ms).

    Returns:
        List of (start_time, end_time) tuples.
    """
    import tensorrt as trt

    cuda_lib = _get_cuda_lib()

    cuda_malloc = cuda_lib.cudaMalloc
    cuda_malloc.restype = ctypes.c_int
    cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

    cuda_memcpy = cuda_lib.cudaMemcpy
    cuda_memcpy.restype = ctypes.c_int
    cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_int]

    cuda_free = cuda_lib.cudaFree
    cuda_free.restype = ctypes.c_int
    cuda_free.argtypes = [ctypes.c_void_p]

    cuda_sync = cuda_lib.cudaDeviceSynchronize
    cuda_sync.restype = ctypes.c_int

    context = engine.create_execution_context()

    # Allocate buffers
    device_ptrs = {}
    buffer_list = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_data = np.random.randn(*shape).astype(dtype)
        nbytes = host_data.nbytes

        ptr = ctypes.c_void_p()
        cuda_malloc(ctypes.byref(ptr), nbytes)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cuda_memcpy(ptr, host_data.ctypes.data, nbytes, MEMCPY_H2D)

        context.set_tensor_address(name, ptr.value)
        device_ptrs[name] = ptr.value
        buffer_list.append(ptr.value)

    cuda_sync()

    # Warmup
    logger.info("Warming up (%d runs)...", warmup)
    for _ in range(warmup):
        for _ in range(iterations_per_run):
            context.execute_v2(buffer_list)
    cuda_sync()

    # Timed runs
    logger.info("Running %d profiling runs (%d iterations each)...",
                num_runs, iterations_per_run)
    timestamps = []
    for i in range(num_runs):
        cuda_sync()
        start = time.monotonic()
        for _ in range(iterations_per_run):
            context.execute_v2(buffer_list)
        cuda_sync()
        end = time.monotonic()
        timestamps.append((start, end))
        time.sleep(0.02)  # Gap for power to settle

    # Cleanup
    for ptr_val in device_ptrs.values():
        cuda_free(ctypes.c_void_p(ptr_val))
    del context

    return timestamps

def build_engine_for_batch_size(onnx_path: str, batch_size: int):
    """Build a TensorRT engine for a specific batch size.

    Overrides the ONNX model's batch dimension to the requested size.
    This works even when the ONNX model has a fixed batch=1.

    Args:
        onnx_path: Path to ONNX model.
        batch_size: Desired batch size.

    Returns:
        TensorRT engine configured for the specified batch size.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Loading ONNX model: %s (batch=%d)", onnx_path, batch_size)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"Failed to parse ONNX: {errors}")

    # Override batch dimension for all inputs
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        shape[0] = batch_size
        inp.shape = tuple(shape)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    logger.info("Building TensorRT engine (batch=%d)...", batch_size)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"Failed to build TensorRT engine for batch={batch_size}")

    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    return engine


def run_trt_inference_batch(engine, batch_size: int, num_runs: int,
                            warmup: int = 5, iterations_per_run: int = 1):
    """Run TensorRT inference at a specific batch size.

    Args:
        engine: TensorRT engine (must support dynamic batch).
        batch_size: Batch size to use.
        num_runs: Number of profiling runs.
        warmup: Warmup runs.
        iterations_per_run: Iterations per profiling run.

    Returns:
        List of (start_time, end_time) tuples.
    """
    import tensorrt as trt

    cuda_lib = _get_cuda_lib()

    cuda_malloc = cuda_lib.cudaMalloc
    cuda_malloc.restype = ctypes.c_int
    cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

    cuda_memcpy = cuda_lib.cudaMemcpy
    cuda_memcpy.restype = ctypes.c_int
    cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_int]

    cuda_free = cuda_lib.cudaFree
    cuda_free.restype = ctypes.c_int
    cuda_free.argtypes = [ctypes.c_void_p]

    cuda_sync = cuda_lib.cudaDeviceSynchronize
    cuda_sync.restype = ctypes.c_int

    context = engine.create_execution_context()

    # Set batch size for all tensors
    device_ptrs = {}
    buffer_list = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))

        # Replace -1 (dynamic) dimension with actual batch size
        shape[0] = batch_size
        context.set_input_shape(name, tuple(shape)) if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(context.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_data = np.random.randn(*shape).astype(dtype)
        nbytes = host_data.nbytes

        ptr = ctypes.c_void_p()
        cuda_malloc(ctypes.byref(ptr), nbytes)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cuda_memcpy(ptr, host_data.ctypes.data, nbytes, MEMCPY_H2D)

        context.set_tensor_address(name, ptr.value)
        device_ptrs[name] = ptr.value
        buffer_list.append(ptr.value)

    cuda_sync()

    # Warmup
    for _ in range(warmup):
        for _ in range(iterations_per_run):
            context.execute_v2(buffer_list)
    cuda_sync()

    # Timed runs
    timestamps = []
    for i in range(num_runs):
        cuda_sync()
        start = time.monotonic()
        for _ in range(iterations_per_run):
            context.execute_v2(buffer_list)
        cuda_sync()
        end = time.monotonic()
        timestamps.append((start, end))
        time.sleep(0.02)

    for ptr_val in device_ptrs.values():
        cuda_free(ctypes.c_void_p(ptr_val))
    del context

    return timestamps
