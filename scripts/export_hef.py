#!/usr/bin/env python

import os
import json
import yaml
import numpy as np
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import Dims


def export_har(ckpt_path: str, hw_arch: str = "hailo8l") -> str:
    """
    Export the trained FingerNet model to Hailo HAR format.

    Args:
        ckpt_path: Path to the checkpoint folder containing the ONNX model.
        hw_arch: Hardware architecture for the Hailo device (default is "hailo8l").

    Returns:
        str: Path to the exported HAR file.
    """
    
    model_name = ckpt_path.split("/")[-2]  # Extract model name from the checkpoint path

    # Set the path for the ONNX model
    onnx_path = os.path.join(ckpt_path, f"{model_name}.onnx")

    # Check if the ONNX model exists
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"The ONNX model {onnx_path} does not exist.")

    # Initialize the Hailo ClientRunner
    runner = ClientRunner(hw_arch=hw_arch)

    # Load the ONNX model
    print("Loading the ONNX model...")
    runner.translate_onnx_model(
        model=onnx_path,
        net_name=model_name,
        # start_node_names=["motion"],  # Starting node name for the model
        # # end_node_names=["force", "shape"],  # Ending node names for the model
        # end_node_names=["output"],  # Ending node names for the model
        # net_input_shapes={"motion": [1, 6, 1, 1]},  # Input shape for the model
        # disable_shape_inference=True,  # Disable shape inference for custom input shapes
    )
    hn = runner.get_hn()
    print(json.dumps(hn, indent=2))

    # Save I/O name mapping
    save_io_mapping(hn, ckpt_path, model_name)

    # Export the model to Hailo HAR format
    print("Exporting the model to Hailo HAR format...")
    har_path = os.path.join(ckpt_path, f"{model_name}.har")
    runner.save_har(har_path)

    print(f"Exported the model to {har_path}")

    return har_path


def quantize_har(har_path: str, dataset_path: str) -> None:
    """
    Quantize the HAR model using the provided dataset.

    Args:
        har_path: Path to the HAR file to be quantized.
        dataset: A numpy array containing the dataset for quantization.
        hw_arch: Hardware architecture for the Hailo device (default is "hailo8l").
    """

    # Load the dataset from the provided path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset {dataset_path} does not exist.")
    print(f"Loading dataset from {dataset_path}...")
    dataset = np.load(dataset_path)
    # Select 2000 samples for quantization randomly if the dataset is larger than 2000 samples
    if dataset.shape[0] > 2000:
        dataset = dataset[np.random.choice(dataset.shape[0], 2000, replace=False)]
    # Standardize the dataset
    with open(os.path.join(os.path.dirname(har_path), "hparams.yaml"), "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    mean = hparams["mean"][:dataset.shape[1]]
    std = hparams["std"][:dataset.shape[1]]
    dataset = (dataset - mean) / std
    dataset = dataset.reshape(dataset.shape[0], 1, 1, -1)  # Reshape to [batch_size, channels, height, width]

    # Check if the har_path exists
    if not har_path.endswith(".har"):
        raise ValueError("The provided path must end with .har")
    if not os.path.exists(har_path):
        raise FileNotFoundError(f"The HAR file {har_path} does not exist.")

    # Initialize the Hailo ClientRunner
    print("Loading the HAR model from", har_path)
    runner = ClientRunner(har=har_path)

    # Set the dataset for quantization
    print("Quantizing the model using the provided dataset...")
    alls_lines = [
        "model_optimization_flavor(optimization_level=4, compression_level=0)",
        "resources_param(max_control_utilization=0.6, max_compute_utilization=0.6, max_memory_utilization=0.6)",
        "performance_param(fps=60, compiler_optimization_level=max)",
    ]
    runner.load_model_script("\n".join(alls_lines))
    runner.optimize(dataset)

    # Save the quantized model
    runner.save_har(har_path)

    print(f"Quantized the model and saved to {har_path}")


def save_io_mapping(hn, save_dir, model_name):
    """
    Save the mapping between layer names and original input/output names.
    """
    input_map = {}
    output_map = {}

    for name, layer in hn["layers"].items():
        if layer["type"] == "input_layer":
            input_map[name] = layer.get("original_names", [""])[0]
        elif layer["type"] == "output_layer":
            output_map[layer.get("input", [""])[0]] = layer.get("original_names", [""])[0]

    mapping = {"inputs": input_map, "outputs": output_map}
    map_path = os.path.join(save_dir, f"{model_name}_io_map.json")
    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved I/O name mapping to {map_path}")


def export_hef(har_path: str) -> None:
    """
    Export the quantized HAR model to Hailo HEF format.

    Args:
        har_path: Path to the HAR file to be exported.
        hw_arch: Hardware architecture for the Hailo device (default is "hailo8l").
    """

    print("Exporting the model to Hailo HEF format...")

    # Check if the har_path exists
    if not har_path.endswith(".har"):
        raise ValueError("The provided path must end with .har")
    if not os.path.exists(har_path):
        raise FileNotFoundError(f"The HAR file {har_path} does not exist.")

    # Initialize the Hailo ClientRunner
    runner = ClientRunner(har=har_path)

    # Compile the model to HEF format
    compiled_hef = runner.compile()

    # Save the compiled HEF model
    hef_path = har_path.replace(".har", ".hef")
    with open(hef_path, "wb") as hef_file:
        hef_file.write(compiled_hef)

    print(f"Compiled the model to {hef_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="lightning_logs/FingerNet/")
    parser.add_argument(
        "--hw_arch", type=str, default="hailo8l", help="Hardware architecture for Hailo device"
    )
    parser.add_argument("--data_path", type=str, help="Path to the dataset for quantization (optional)")
    args = parser.parse_args()

    # Export the model to Hailo HAR format
    har_path = export_har(args.ckpt_path, args.hw_arch)

    # If a dataset is provided, quantize the HAR model
    if args.data_path:

        # Quantize the HAR model using the dataset
        quantize_har(har_path, args.data_path)

        # Export the quantized HAR model to HEF format
        export_hef(har_path)
