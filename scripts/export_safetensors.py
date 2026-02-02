#!/usr/bin/env python

"""
Export SafeTensors Model

This script exports a trained NeckNet model to the .safetensors format.
It is used for safe and efficient model sharing on Hugging Face.

Example usage:

```bash
python export_safetensors.py --ckpt_path <checkpoint_path>
```

where <checkpoint_path> is the path to the checkpoint folder.
"""

import argparse
import json
import os
import torch
from safetensors.torch import save_file
from omnineck.models.torch.necknet import NeckNet


def extract_config_from_ckpt(ckpt_path: str) -> dict:
    """
    Load Lightning checkpoint and extract model config for inference.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "hyper_parameters" not in ckpt:
        raise RuntimeError("No hyper_parameters found in checkpoint")

    hp = ckpt["hyper_parameters"]

    config = {
        "architectures": ["NeckNetModel"],
        "model_type": "necknet",
        "framework": "pytorch",
        "library_name": "omnineck",
        "x_dim": hp["x_dim"],
        "y_dim": hp["y_dim"],
        "hidden_dim": hp["hidden_dim"],
        "mean": hp["mean"],
        "std": hp["std"],
    }

    return config


def safetensors_export(ckpt_root: str) -> None:
    device = torch.device("cpu")

    ckpt_dir = os.path.join(ckpt_root, "checkpoints")
    ckpt_file = os.listdir(ckpt_dir)[0]
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    print(f"Loading checkpoint: {ckpt_path}")

    # ---------- load model ----------
    model = NeckNet.load_from_checkpoint(
        ckpt_path,
        map_location=device,
    )
    model.eval()

    # ---------- export safetensors ----------
    state_dict = model.state_dict()

    if not os.path.exists(os.path.join(ckpt_root, "safetensors")):
        os.makedirs(os.path.join(ckpt_root, "safetensors"))
    output_model_path = os.path.join(ckpt_root, "safetensors", "model.safetensors")

    metadata = {
        "format": "pt",
        "framework": "pytorch",
        "model_type": "NeckNet",
        "library_name": "omnineck",
    }

    save_file(
        state_dict,
        output_model_path,
        metadata=metadata,
    )

    print(f"Saved model weights to {output_model_path}")

    # ---------- export config.json ----------
    config = extract_config_from_ckpt(ckpt_path)

    output_config_path = os.path.join(ckpt_root, "safetensors", "config.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved config to {output_config_path}")

    # ---------- size ----------
    size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the checkpoint folder.",
    )
    args = parser.parse_args()

    safetensors_export(args.ckpt_path)
