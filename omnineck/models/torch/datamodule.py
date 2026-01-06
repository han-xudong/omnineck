#!/usr/bin/env python

import os
import numpy as np
import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule


class NeckDataset(Dataset):
    """OmniNeck dataset.

    OmniNeck dataset sample contains:
        - pose: 6-dim
        - force: 6-dim
        - shape: 3n-dim
    """

    def __init__(self, data: np.ndarray, transform=None):
        """Initialize the dataset.

        Args:
            data (np.ndarray): The dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data = data
        self.transform = transform

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """Get the item of the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: The item of the dataset.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_item = self.data[idx, :].astype("float32").reshape(-1, self.data.shape[1])
        data_tensor = torch.from_numpy(data_item)
        return data_tensor


class NeckDataModule(LightningDataModule):
    """OmniNeck data module.

    The dataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader)
        - test_dataloader (the test dataloader)
    """

    def __init__(
        self,
        data_folder: str,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_val_split: Tuple[float, float] = (0.875, 0.125),
    ):
        """Initialize the data module.

        Args:
            train_val_split (Tuple[float, float], optional): The train/val split. Defaults to (0.875, 0.125).
            batch_size (int, optional): The batch size. Defaults to 128.
            num_workers (int, optional): The number of workers. Defaults to 4.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
        """

        super().__init__()
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_folder = data_folder

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """

        pass

    def setup(self, stage: Optional[str] = None):
        """Load data.

        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`.

        Args:
            stage (Optional[str], optional): The stage. Defaults to None.
        """

        if not self.data_train or not self.data_val or not self.data_test:

            data_path = self.data_folder + "train_data.npy"
            if not os.path.exists(data_path):
                raise ValueError("Data file does not exist.")

            data = np.load(data_path)

            dataset = NeckDataset(data=data, transform=None)

            train_length = int(self.train_val_split[0] * len(dataset.data))
            val_length = int(len(dataset.data) - train_length)

            self.data_train, self.data_val = random_split(
                dataset,
                (train_length, val_length),
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )
