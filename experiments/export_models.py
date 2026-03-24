#!/usr/bin/env python3
"""
Export 5 models to ONNX with STATIC shapes for TensorRT.
"""

import torch
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

INPUT_SHAPE = (1, 3, 224, 224)


def export_model(name, model, filename):
    """Export with static batch size = 1."""
    model.eval()
    dummy = torch.randn(*INPUT_SHAPE)

    output_path = MODELS_DIR / filename

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # NO dynamic_axes — fully static shapes
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    params = sum(p.numel() for p in model.parameters())

    print(f"  {name:20s} → {filename:30s} "
          f"({size_mb:.1f} MB, {params/1e6:.1f}M params)")

    return size_mb, params


def main():
    from torchvision import models

    print("Exporting models to ONNX (static shapes)...\n")

    results = {}

    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    results["mobilenetv2"] = export_model(
        "MobileNetV2", m, "mobilenetv2.onnx"
    )

    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    results["resnet18"] = export_model(
        "ResNet-18", m, "resnet18.onnx"
    )

    m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    results["resnet34"] = export_model(
        "ResNet-34", m, "resnet34.onnx"
    )

    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    results["resnet50"] = export_model(
        "ResNet-50", m, "resnet50.onnx"
    )

    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    results["efficientnet_b0"] = export_model(
        "EfficientNet-B0", m, "efficientnet_b0.onnx"
    )

    print(f"\nAll models exported to {MODELS_DIR}/")
    print(f"\nSummary for paper Table III:")
    print(f"{'Model':20s} {'Params (M)':>12s} {'ONNX Size (MB)':>16s}")
    print("-" * 50)
    for name, (size_mb, params) in results.items():
        print(f"{name:20s} {params/1e6:>12.1f} {size_mb:>16.1f}")

    print("\nSanity check:")
    for name, (size_mb, params) in results.items():
        if size_mb < 1.0:
            print(f"  WARNING: {name} is only {size_mb:.1f} MB")
        else:
            print(f"  OK: {name} = {size_mb:.1f} MB (static shape)")


if __name__ == "__main__":
    main()