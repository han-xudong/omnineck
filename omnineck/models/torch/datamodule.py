"""
OmniNeck DataModule.

This module implements the OmniNeck DataModule for PyTorch Lightning.
It provides data loading and preprocessing functionalities for training
and validation of the OmniNeck model.

Usage:

```python
from omnineck.models import OmniNeckDataModule
datamodule = OmniNeckDataModule(
    dataset_path=<dataset_path>,
    batch_size=<batch_size>,
    num_workers=<num_workers>,
    pin_memory=<pin_memory>,
    train_val_split=<train_val_split>,
)
datamodule.setup()
```

where `<dataset_path>` is the path to the dataset, `<batch_size>` is the batch size,
`<num_workers>` is the number of workers for data loading, `<pin_memory>` is a boolean
indicating whether to pin memory, and `<train_val_split>` is a tuple indicating the
train/validation split ratios.
"""

import os
import h5py
import numpy as np
import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule


class OmniNeckDataset(Dataset):
    """
    OmniNeck dataset.
    """

    def __init__(self, data: np.ndarray, transform=None):
        """
        Initialize the dataset.

        Args:
            data (np.ndarray): The dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data = data
        self.transform = transform

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the item of the dataset.

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


class OmniNeckDataModule(LightningDataModule):
    """
    OmniNeck data module.
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        train_val_split: Tuple[float, float] = (0.875, 0.125),
    ) -> None:
        """
        Initialize the data module.

        Args:
            data_folder (str): The folder containing the dataset.
            batch_size (int, optional): The batch size. Defaults to 128.
            num_workers (int, optional): The number of workers. Defaults to 4.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
            train_val_split (Tuple[float, float], optional): The train/val split. Defaults to (0.875, 0.125).
        """

        super().__init__()
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset_path = dataset_path

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train or not self.data_val or not self.data_test:

            data_path = os.path.join(self.dataset_path, "data.h5")
            if not os.path.exists(data_path):
                raise ValueError("Data file does not exist.")

            with h5py.File(data_path, "r") as f:
                pose = f["train/pose"][:]
                force = f["train/force"][:]
                surface_node = f["train/surface_node"][:, :, 1:]

            data = np.concatenate([pose, force, surface_node.reshape(surface_node.shape[0], -1)], axis=1)

            dataset = OmniNeckDataset(data=data, transform=None)

            train_length = int(self.train_val_split[0] * len(dataset.data))
            val_length = int(len(dataset.data) - train_length)

            self.data_train, self.data_val = random_split(
                dataset,
                (train_length, val_length),
                generator=torch.Generator().manual_seed(42),
            )

            # calculate mean and std from training data
            all_train_data = torch.cat([self.data_train[i] for i in range(len(self.data_train))], dim=0)
            self.data_mean = torch.mean(all_train_data, dim=0)
            self.data_std = torch.std(all_train_data, dim=0)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=True,
        )
