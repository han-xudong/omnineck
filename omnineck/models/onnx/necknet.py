#!/usr/bin/env python

"""
NeckNet Inference

NeckNet is a neural network model for infering the proprioception of the soft OmniNeck.
The input is the motion, and the output is the force on the bottom surface and the displacement of mesh nodes.
The model is implemented in PyTorch and exported to ONNX format for inference.

Example usage:
```bash
python necknet.py --onnx_path ./models/NeckNet.onnx
```

For more information, please refer to https://github.com/han-xudong/omnineck
"""

import argparse
from typing import Tuple
import numpy as np
from omnineck.utils.nn_utils import init_model

class NeckNet:
    def __init__(self, onnx_path) -> None:
        """NeckNet initialization.

        Args:
            onnx_path: The path of the model.
        """
        
        # Create a ONNX runtime model
        try:
            self.model = init_model(onnx_path)
        except Exception as e:
            raise ValueError(f"Failed to load the model: {e}")

        # Print the initialization message
        print("Model Path:", onnx_path)
        print(
            "Input:",
            [f"{input.name} ({input.shape[0]}, {input.shape[1]})" for input in self.model.get_inputs()],
        )
        print(
            "Output:",
            [f"{output.name} ({output.shape[0]}, {output.shape[1]})" for output in self.model.get_outputs()],
        )
     
    def infer(self, motion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inference.

        Args:
            motion (np.ndarray): The motion of the OmniNeck.
            
        Returns:
            force (np.ndarray): The force on the bottom surface of the OmniNeck.
            node (np.ndarray): The node displacement of the OmniNeck.
        """

        return self.model.run(None, {"motion": motion.astype(np.float32).reshape(1, -1)})
    

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="NeckNet inference.")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./models/NeckNet.onnx",
        help="The path of the model.",
    )
    args = parser.parse_args()
    
    # Initialize the NeckNet
    necknet = NeckNet(args.onnx_path)