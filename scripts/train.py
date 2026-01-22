#!/usr/bin/env python

"""
Training script for NeckNet.

Usage:

```bash
python scripts/train.py
```

Various configuration options are available:

╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
| Options               | Description                                           | Type   | Default                      |
|-----------------------|-------------------------------------------------------|--------|------------------------------|
| --batch-size          | Batch size for training.                              | int    | 128                          |
| --lr                  | Learning rate for the optimizer.                      | float  | 1e-5                         |
| --max-epochs          | Maximum number of training epochs.                    | int    | 2000                         |
| --save-dir            | Directory to save training logs and checkpoints.      | str    | lightning_logs               |
| --zero-loss-weight    | Weight for the zero loss component.                   | float  | 0.5                          |
| --data.dataset-path   | Path to the dataset directory.                        | str    | ./data/omnineck/sim          |
| --data.num-workers    | Number of workers for data loading.                   | int    | 4                            |
| --data.pin-memory     | Whether to pin memory during data loading.            | bool   | False                        |
| --data.train-val-split| Train-validation split ratios.                        | tuple  | 0.875 0.125                  |
| --model.name          | Model name.                                           | str    | NeckNet                      |
| --model.x-dim         | Input dimension.                                      | tuple  | 6                            |
| --model.y-dim         | Output dimension.                                     | tuple  | 6 2862                       |
| --model.hidden-dim    | Hidden layer dimensions for each part of the model.   | tuple  | 512 512 512 1024 1024 1024   |
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
"""

import os
import time
import tyro
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omnineck.models import OmniNeckDataModule
from omnineck.models import NeckNet
from omnineck.configs.train import TrainConfig


def main(cfg: TrainConfig) -> None:
    """
    Training function.

    Args:
        cfg (TrainConfig): The training configuration.
    """

    # DataModule
    datamodule = OmniNeckDataModule(
        dataset_path=cfg.data.dataset_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_val_split=cfg.data.train_val_split,
    )
    datamodule.setup()

    # Callbacks
    logger = TensorBoardLogger(
        save_dir=cfg.save_dir,
        name=cfg.model.name,
        version=time.strftime("%m%d-%H%M"),
    )
    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch:04d}-{val_loss:.4f}",
        save_last=True,
    )
    trainer = Trainer(
        logger=logger,
        accelerator="auto",
        max_epochs=cfg.max_epochs,
        callbacks=[
            model_checkpoint_callback,
            LearningRateMonitor("epoch"),
        ],
        gradient_clip_algorithm="norm",
        gradient_clip_val=1e3,
    )

    model = NeckNet(
        x_dim=list(cfg.model.x_dim),
        y_dim=list(cfg.model.y_dim),
        hidden_dim=[list(h) for h in cfg.model.hidden_dim],
        mean=datamodule.data_mean.tolist(),
        std=datamodule.data_std.tolist(),
        zero_loss_weight=cfg.zero_loss_weight,
        lr=cfg.lr,
    )

    # Training
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt:
        raise RuntimeError("No best checkpoint found. Check monitor key.")
    print("Best ckpt:", best_ckpt)

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    torch.save(
        state_dict, os.path.join(trainer.logger.log_dir, f"{cfg.model.name}_{trainer.logger.version}.pt")
    )


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)

    main(cfg=cfg)
