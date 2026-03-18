"""
Create heavy ONNX models for PowerLens demos.

Creates multiple models of different sizes to demonstrate:
- Power scaling with model complexity
- Batch size energy scaling
- Thermal throttling under sustained load
"""

import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Install onnx: pip install onnx")
    exit(1)


def create_model(output_path, num_blocks=8, channels=512, name="heavy"):
    """Create a conv net with configurable depth and width.

    Args:
        output_path: Where to save the ONNX file.
        num_blocks: Number of conv-relu blocks (more = slower).
        channels: Channel width (more = more compute per layer).
        name: Model name for the graph.
    """
    X = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 3, 224, 224]
    )
    Y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 10]
    )

    nodes = []
    initializers = []

    # First conv: 3 -> channels, 7x7, stride 2
    # Output: batch x channels x 112 x 112
    w = numpy_helper.from_array(
        np.random.randn(channels, 3, 7, 7).astype(np.float32) * 0.01,
        name="conv0_w"
    )
    initializers.append(w)
    nodes.append(helper.make_node(
        "Conv", ["input", "conv0_w"], ["conv0_out"],
        kernel_shape=[7, 7], strides=[2, 2], pads=[3, 3, 3, 3]
    ))
    nodes.append(helper.make_node("Relu", ["conv0_out"], ["relu0_out"]))

    prev_output = "relu0_out"
    prev_channels = channels

    # Heavy blocks: channels -> channels, 3x3, stride 1
    # These keep the spatial dimensions large = lots of compute
    for i in range(1, num_blocks + 1):
        w_name = f"conv{i}_w"
        conv_out = f"conv{i}_out"
        relu_out = f"relu{i}_out"

        w = numpy_helper.from_array(
            np.random.randn(channels, prev_channels, 3, 3).astype(np.float32) * 0.01,
            name=w_name
        )
        initializers.append(w)

        nodes.append(helper.make_node(
            "Conv", [prev_output, w_name], [conv_out],
            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]
        ))
        nodes.append(helper.make_node("Relu", [conv_out], [relu_out]))

        prev_output = relu_out
        prev_channels = channels

    # Downsample conv: stride 2 to reduce spatial dims
    # Output: batch x channels x 56 x 56
    w = numpy_helper.from_array(
        np.random.randn(channels, channels, 3, 3).astype(np.float32) * 0.01,
        name="conv_down_w"
    )
    initializers.append(w)
    nodes.append(helper.make_node(
        "Conv", [prev_output, "conv_down_w"], ["conv_down_out"],
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]
    ))
    nodes.append(helper.make_node("Relu", ["conv_down_out"], ["relu_down_out"]))

    # Global average pool
    nodes.append(helper.make_node(
        "GlobalAveragePool", ["relu_down_out"], ["gap_out"]
    ))

    # Flatten
    nodes.append(helper.make_node(
        "Flatten", ["gap_out"], ["flat_out"], axis=1
    ))

    # FC: channels -> 10
    fc_w = numpy_helper.from_array(
        np.random.randn(10, channels).astype(np.float32) * 0.01,
        name="fc_w"
    )
    fc_b = numpy_helper.from_array(
        np.zeros(10).astype(np.float32),
        name="fc_b"
    )
    initializers.extend([fc_w, fc_b])
    nodes.append(helper.make_node(
        "Gemm", ["flat_out", "fc_w", "fc_b"], ["output"],
        transB=1
    ))

    graph = helper.make_graph(
        nodes, name, [X], [Y], initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  {output_path}: {num_blocks} blocks, {channels}ch, {size_mb:.1f}MB")


if __name__ == "__main__":
    print("Creating PowerLens demo models...")
    print()

    # Light model: fast inference, low power
    create_model("demo_light.onnx", num_blocks=2, channels=64, name="light")

    # Medium model: moderate load
    create_model("demo_medium.onnx", num_blocks=4, channels=256, name="medium")

    # Heavy model: high GPU load, should cause thermal rise
    create_model("demo_heavy.onnx", num_blocks=8, channels=512, name="heavy")

    print()
    print("Done! Use with:")
    print("  powerlens profile --onnx demo_heavy.onnx")
    print("  powerlens compare demo_light.onnx demo_heavy.onnx")