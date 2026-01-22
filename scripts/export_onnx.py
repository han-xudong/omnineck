#!/usr/bin/env python

"""
Export ONNX Model

This script is to export the trained model to ONNX format.

Example usage:

```bash
python export_onnx.py --ckpt_dir <ckpt_dir>
```

where <ckpt_dir> is the path to the checkpoint folder.
"""

import argparse
import os
import torch
import onnx
from omnineck.models import NeckNet


def onnx_export(ckpt_dir: str) -> None:
    """
    Export the trained model to ONNX format.

    Args:
        ckpt_dir (str): Path to the checkpoint folder.
    """
    
    if not ckpt_dir.endswith("/"):
        ckpt_dir += "/"
    model_name = "_".join(os.path.join("", ckpt_dir).split("/")[1:-1])
    print(f"Exporting {model_name} model to ONNX format")

    ckpt_path = os.path.join(ckpt_dir, "checkpoints", os.listdir(os.path.join(ckpt_dir, "checkpoints"))[0])
    # Load the model
    device = torch.device("cpu")
    model = NeckNet.load_from_checkpoint(ckpt_path, weights_only=False).to(device)
    model.eval()

    # Get input dimension from the model
    input_dim = model.x_dim[0]  # Use the first input dimension

    # Export the model
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
    onnx_path = os.path.join(ckpt_dir, "model.onnx")

    # Export with only one input
    torch.onnx.export(
        model,
        dummy_input,  # Only the motion input
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["motion"],
        output_names=["force", "shape"],
        dynamic_axes={"motion": {0: "batch_size"}, "force": {0: "batch_size"}, "shape": {0: "batch_size"}},
    )

    print(f"Exported the model to {onnx_path}")

    # Check the exported model
    onnx_model = onnx.load(onnx_path)

    print("\n=== Model Inputs ===")
    for input_tensor in onnx_model.graph.input:
        name = input_tensor.name
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        dtype = input_tensor.type.tensor_type.elem_type
        print(f"Name: {name}, Shape: {shape}, Type: {dtype}")
    print("\n=== Model Outputs ===")
    for output_tensor in onnx_model.graph.output:
        name = output_tensor.name
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        dtype = output_tensor.type.tensor_type.elem_type
        print(f"Name: {name}, Shape: {shape}, Type: {dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Path to the checkpoint folder.",
    )
    args = parser.parse_args()
    onnx_export(args.ckpt_dir)
