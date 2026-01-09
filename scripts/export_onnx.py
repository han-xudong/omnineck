#!/usr/bin/env python

"""
Export ONNX Model

This script is to export the trained FingerNet model to ONNX format.

Example usage:

```bash
python export_onnx.py --ckpt_path <checkpoint_path>
```

where <checkpoint_path> is the path to the checkpoint folder.
"""

import argparse
import os
import torch
from torch import Tensor
from typing import List
import onnx
from omnineck.models import NeckNet


class NeckNetRuntime(NeckNet):
    """
    NeckNetRuntime is a NeckNet model for runtime inference.
    It is used for ONNX export and real-world deployment.
    """
    
    def __init__(
        self,
        x_dim: list,
        y_dim: list,
        h1_dim: list,
        h2_dim: list,
        **kwargs,
    ) -> None:
        """
        Initialize the model.
        
        Args:
            x_dim: dimension of the input data
            y_dim: dimension of the output data
            h1_dim: dimension of the hidden layer 1
            h2_dim: dimension of the hidden layer 2
        """
        
        # Call the super constructor
        super().__init__(x_dim, y_dim, h1_dim, h2_dim, **kwargs)
            
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor.
        
        Returns:
            Tuple of output tensors.
        """
        
        outputs = []
        for i in range(len(self.y_dim)):
            y = getattr(self, f"estimator_{i}")(x)
            outputs.append(y)
        return outputs

def onnx_export(ckpt_path: str) -> None:
    # Load the model
    device = torch.device("cpu")
    model = NeckNetRuntime.load_from_checkpoint(
        os.path.join(
            ckpt_path + "checkpoints", os.listdir(ckpt_path + "checkpoints")[0]
        ),
    ).to(device)
    model.eval()
    
    model_name = ckpt_path.split("/")[-3]  # Extract model name from the path
    print(f"Exporting {model_name} model to ONNX format...")

    # Get input dimension from the model
    input_dim = model.x_dim[0]  # Use the first input dimension
    
    # Export the model
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
    onnx_path = os.path.join(ckpt_path, f"{model_name}.onnx")
    
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
        dynamic_axes={'motion': {0: 'batch_size'}, 'force': {0: 'batch_size'}, 'shape': {0: 'batch_size'}},
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
    parser.add_argument("--ckpt_path", type=str, default="lightning_logs/NeckNet/0108-1949/", help="Path to the checkpoint folder.")
    args = parser.parse_args()
    onnx_export(args.ckpt_path)
