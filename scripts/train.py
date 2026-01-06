#!/usr/bin/env python

"""
Training script for NeckNet.

The training configuration is from './config/training.yaml'.
The log files will be saved in the './lightning_logs' folder.
"""

import argparse
import os
import time
import shutil
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from omnineck.models import NeckDataModule
from omnineck.models import NeckNet


# Set the training function
def main(
    lr: float = 1e-4,
    batch_size: int = 128,
    max_epochs: int = 2000,
) -> None:
    """
    Training function.

    The training configuration is from './config/training.yaml'.
    The log files will be saved in the './lightning_logs' folder.

    Args:
        z_dim: the dimension of the latent code
        recon_pred_scale: scale for the reconstruction and prediction loss
        kl_coef: coefficient for the KL divergence
        z_coef: coefficient for the latent loss
    """

    with open("./config/training.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    x_dim = config["x_dim"]
    y_dim = config["y_dim"]
    h1_dim = config["h1_dim"]
    h2_dim = config["h2_dim"]
    data_folder = config["data_folder"]

    # DataModule
    dm = NeckDataModule(
        batch_size=batch_size,
        num_workers=8,
        data_folder=data_folder,
    )
    dm.setup()

    # Callbacks
    trainer = Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
        gradient_clip_algorithm="norm",
        gradient_clip_val=1e3,
    )

    model = NeckNet(
        x_dim=x_dim,
        y_dim=y_dim,
        h1_dim=h1_dim,
        h2_dim=h2_dim,
        lr=lr,
    )

    # Training
    try:
        trainer.fit(model, dm)
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Evaluation
    model.eval()

    # Rename log folder
    if trainer.logger is not None and hasattr(trainer.logger, "version"):
        version_folder = "./lightning_logs/version_" + str(trainer.logger.version)
    else:
        # Fallback to latest version folder if logger is None
        logs = [d for d in os.listdir("./lightning_logs") if d.startswith("version_")]
        if logs:
            version_folder = os.path.join(
                "./lightning_logs", sorted(logs, key=lambda x: int(x.split("_")[-1]))[-1]
            )
        else:
            raise RuntimeError("No version folder found in ./lightning_logs.")

    if type == "conv":
        new_version_folder = "./lightning_logs/ConvNeckNet_" + time.strftime("%m%d-%H%M")
    else:
        # Default to NeckNet for other types
        new_version_folder = "./lightning_logs/NeckNet_" + time.strftime("%m%d-%H%M")
    if os.path.exists(new_version_folder):
        shutil.rmtree(new_version_folder)
    os.rename(version_folder, new_version_folder)
    # Save the model as pt
    model_path = os.path.join(new_version_folder, "NeckNet.pt")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128).")
    parser.add_argument("--max-epochs", type=int, default=2000, help="Max epochs (default: 2000).")
    args = parser.parse_args()

    main(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
