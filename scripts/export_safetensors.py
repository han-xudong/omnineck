#!/usr/bin/env python

"""
Export SafeTensors Model

This script exports a trained FingerNet model to the .safetensors format.
It is used for safe and efficient model sharing on Hugging Face.

Example usage:

```bash
python export_safetensors.py --ckpt_path <checkpoint_path>
```

where <checkpoint_path> is the path to the checkpoint folder.
"""

import argparse
import os
import torch
from safetensors.torch import save_file
from metafinger.models.torch.fingernet import FingerNet


def safetensors_export(ckpt_path: str) -> None:
    device = torch.device("cpu")
    ckpt_dir = os.path.join(ckpt_path, "checkpoints")
    ckpt_file = os.listdir(ckpt_dir)[0]
    ckpt_full_path = os.path.join(ckpt_dir, ckpt_file)

    print(f"Loading checkpoint: {ckpt_full_path}")
    model = FingerNet.load_from_checkpoint(ckpt_full_path, map_location=device)
    model.eval()

    model_name = ckpt_path.rstrip("/").split("/")[-1]
    output_path = os.path.join(ckpt_path, f"{model_name}.safetensors")

    state_dict = model.state_dict()

    save_file(state_dict, output_path)
    print(f"Exported {model_name} weights to {output_path}")

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="lightning_logs/fingernet_surf/",
        help="Path to the checkpoint folder.",
    )
    args = parser.parse_args()
    safetensors_export(args.ckpt_path)
