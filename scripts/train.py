#!/usr/bin/env python

"""
Training script for NeckNet.

Usage:

```bash
uv run python scripts/train.py
```

Various configuration options are available:
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
| Options               | Description                                       | Type   | Default               |
|-----------------------|---------------------------------------------------|--------|-----------------------|
| --batch-size          | Batch size for training.                          | int    | 128                   |
| --lr                  | Learning rate for the optimizer.                  | float  | 1e-5                  |
| --max-epochs          | Maximum number of training epochs.                | int    | 2000                  |
| --save-dir            | Directory to save training logs and checkpoints.  | str    | lightning_logs        |
|--data.dataset-path    | Path to the dataset directory.                    | str    | ./data/omnineck/sim   |
|--data.num-workers     | Number of workers for data loading.               | int    | 4                     |
|--data.pin-memory      | Whether to pin memory during data loading.        | bool   | False                 |
|--data.train-val-split | Train-validation split ratios.                    | tuple  | 0.875 0.125           |
|--model.name           | Model name                                        | str    | NeckNet               |
|--model.x-dim          | Input dimension                                   | tuple  | 6                     |
|--model.y-dim          | Output dimension                                  | tuple  | 6 1800                |
|--model.h1-dim         | Hidden layer 1 dimension                          | tuple  | 128 1024              |
|--model.h2-dim         | Hidden layer 2 dimension                          | tuple  | 128 1024              |
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
"""

import os
import time
import tyro
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omnineck.models import NeckDataModule
from omnineck.models import NeckNet
from omnineck.configs.train import TrainConfig

# Set the training function
def main(
    cfg: TrainConfig,
) -> None:
    """
    Training function.

    Args:
        cfg (TrainConfig): The training configuration.
    """

    # DataModule
    datamodule = NeckDataModule(
        dataset_path=cfg.data.dataset_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_val_split=cfg.data.train_val_split,
    )

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
        filename="epoch{epoch:04d}-val_loss{val_loss:.4f}",
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
        x_dim=cfg.model.x_dim,
        y_dim=cfg.model.y_dim,
        h1_dim=cfg.model.h1_dim,
        h2_dim=cfg.model.h2_dim,
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

    ckpt = torch.load(best_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]

    torch.save(state_dict, os.path.join(trainer.logger.log_dir, f"{cfg.model.name}.pt"))


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    
    main(cfg=cfg)
