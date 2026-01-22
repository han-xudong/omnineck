#!/usr/bin/env python

from typing import Any, List, Tuple
import numpy as np
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class Normalizer(nn.Module):
    def __init__(self, mean: Tensor, std: Tensor, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: Tensor) -> Tensor:
        return x * (self.std + self.eps) + self.mean


class NeckNet(LightningModule):
    """
    NeckNet is a PyTorch Lightning model for the omni-neck dataset.
    It is used for training and evaluation of the model.
    """

    def __init__(
        self,
        x_dim: list,
        y_dim: list,
        hidden_dim: list,
        mean: list = None,
        std: list = None,
        zero_loss_weight: float = 1.0,
        lr: float = 1e-4,
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
        super().__init__(**kwargs)

        # Set the model parameters
        self.save_hyperparameters()
        self.lr = lr
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.zero_loss_weight = zero_loss_weight
        
        # Check dimensions
        assert len(self.x_dim) == 1, "Only one input dimension is supported."
        assert len(self.y_dim) == len(self.hidden_dim), "Output and hidden dimensions must match."
        assert len(mean) == sum(self.x_dim) + sum(self.y_dim), "Mean dimension mismatch."
        assert len(std) == sum(self.x_dim) + sum(self.y_dim), "Std dimension mismatch."

        self.x_mean = []
        self.x_std = []
        self.y_mean = []
        self.y_std = []

        if mean is not None and std is not None:
            data_dim = np.concatenate((self.x_dim, self.y_dim))
            data_start = 0
            data_end = 0
            for i in range(len(data_dim)):
                data_end += data_dim[i]
                if i < len(self.x_dim):
                    self.x_mean.append(mean[data_start:data_end])
                    self.x_std.append(std[data_start:data_end])
                else:
                    self.y_mean.append(mean[data_start:data_end])
                    self.y_std.append(std[data_start:data_end])
                data_start = data_end
        else:
            assert False, "Mean and std must be provided."

        # Define the model architecture
        self.x_normalizers = nn.ModuleList(
            [
                Normalizer(
                    mean=torch.tensor(self.x_mean[i], dtype=torch.float32),
                    std=torch.clamp(torch.tensor(self.x_std[i], dtype=torch.float32), min=1e-8),
                )
                for i in range(len(self.x_dim))
            ]
        )
        
        self.y_normalizers = nn.ModuleList(
            [
                Normalizer(
                    mean=torch.tensor(self.y_mean[i], dtype=torch.float32),
                    std=torch.clamp(torch.tensor(self.y_std[i], dtype=torch.float32), min=1e-8),
                )
                for i in range(len(self.y_dim))
            ]
        )
        
        self.estimators = nn.ModuleList()
        for i in range(len(self.y_dim)):
            layers = []
            in_dim = self.x_dim[0]

            for j in range(len(self.hidden_dim[i])):
                out_dim = self.hidden_dim[i][j]
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim

            layers.append(nn.Linear(in_dim, self.y_dim[i]))

            self.estimators.append(nn.Sequential(*layers))


    @staticmethod
    def pretrained_weights_available():
        """Check if the pretrained weights are available.

        Returns:
            bool: True if the pretrained weights are available, False otherwise.
        """

        pass

    def from_pretrained(self, checkpoint_name):
        """Load the pretrained weights.

        Args:
            checkpoint_name: the name of the checkpoint file.

        Returns:
            None
        """

        pass

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward pass of the model.

        Args:
            x: the input data.

        Returns:
            y_list: the predicted data.
        """

        x = self.x_normalizers[0].normalize(x)

        y_list = []
        for i in range(len(self.y_dim)):
            y = self.estimators[i](x)
            y = self.y_normalizers[i].denormalize(y)
            y_list.append(y)

        # Return the output data
        return y_list

    def forward_with_index(self, x: Tensor, output_index: int) -> Tensor:
        """Forward pass of the model with index.

        Args:
            x: the input data.
            output_index: the index of the output layer.

        Returns:
            x_recon: the reconstructed data.
            mu: the mu.
            std: the std.
        """

        x = self.x_normalizers[0].normalize(x)

        y = self.estimators[output_index](x)
        y = self.y_normalizers[output_index].denormalize(y)

        # Return the output data
        return y

    def _prepare_batch(self, batch: Any) -> Tensor:
        """Prepare the batch.

        Args:
            batch: the input batch.

        Returns:
            x: the input batch.
        """

        # Reshape the batch
        x = batch
        return x.view(x.size(0), -1)

    def step(self, batch: Any, batch_idx: int) -> Tuple[Tensor, dict]:
        """Step the model.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
            logs: the logs.
        """

        # Prepare the batch
        data = self._prepare_batch(batch)

        # Split the data
        data_list = []
        data_start = 0
        data_end = 0
        data_dim = np.concatenate((self.x_dim, self.y_dim))
        for i in range(len(data_dim)):
            data_end += data_dim[i]
            data_i = data[:, data_start:data_end]
            data_list.append(data_i)
            data_start = data_end

        # Define the logs
        logs = {}

        # Define the loss
        loss_list = []
        zero_loss_list = []
        
        # Calculate the prediction loss
        for i in range(len(self.y_dim)):
            y_i = data_list[i + 1]
            y_i_norm = self.y_normalizers[i].normalize(y_i)
            x_norm = self.x_normalizers[0].normalize(data_list[0])
            y_hat_i_norm = self.estimators[i](x_norm)
            loss = F.mse_loss(y_hat_i_norm, y_i_norm)
            loss_list.append(loss)
            logs[f"loss_{i}"] = loss
        
        # Calculate the zero loss
        for i in range(len(self.y_dim)):
            y_i_zero = torch.zeros_like(data_list[i + 1])
            y_i_zero_norm = self.y_normalizers[i].normalize(y_i_zero)
            x_zero_norm = self.x_normalizers[0].normalize(torch.zeros_like(data_list[0]))
            y_hat_i_zero_norm = self.estimators[i](x_zero_norm)
            zero_loss = F.mse_loss(y_hat_i_zero_norm, y_i_zero_norm)
            zero_loss_list.append(zero_loss)
            logs[f"zero_loss_{i}"] = zero_loss

        # Calculate the total loss
        loss = sum(loss_list) + self.zero_loss_weight * sum(zero_loss_list)

        logs["loss"] = loss
        loss = Tensor(loss)

        # Return the loss and logs
        return loss, logs

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
        """

        # Run the step
        loss, logs = self.step(batch, batch_idx)
        # Log the data
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)

        # Return the loss
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Validation step.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
        """

        # Run the step
        loss, logs = self.step(batch, batch_idx)
        # Log the data
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)

        # Return the loss
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        """Test step.

        Args:
            batch: the input batch.
            batch_idx: the batch index.

        Returns:
            loss: the loss.
        """

        pass

    def test_epoch_end(self, outputs: List[Any]):
        """Test epoch end.

        Args:
            outputs: the outputs.

        Returns:
            None
        """

        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizers.

        Returns:
            optimizer: the optimizer.
        """

        # Initialize a list to hold the parameters to optimize
        params = []

        # Iterate over each output branch (estimator)
        for i in range(len(self.y_dim)):
            # Add the parameters of the estimator to the optimizer's parameter list
            params += list(self.estimators[i].parameters())

        # Filter out parameters that have requires_grad set to False
        params = list(filter(lambda p: p.requires_grad, params))

        # Return the Adam optimizer configured with the filtered parameters
        return torch.optim.Adam(params, lr=self.lr)

    def export_torchscript(self, path):
        """Export the model to TorchScript format.

        Args:
            path: the path to save the exported model.
        """

        # Create a traced script module
        traced = torch.jit.script(self)

        # Save the traced script module
        traced.save(path)
        print(f"Saved model to: {path}")


if __name__ == "__main__":
    # Create the model
    model = NeckNet(x_dim=[6], y_dim=[6, 1800], h1_dim=[100, 1000], h2_dim=[100, 1000], lr=1e-4)
    # Print the model
    print(model)
    # Print the hyperparameters
    print(model.hparams)
