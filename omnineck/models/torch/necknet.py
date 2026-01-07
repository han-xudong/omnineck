#!/usr/bin/env python

from typing import Any, List, Tuple
import numpy as np
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class NeckNet(LightningModule):
    """
    NeckNet is a PyTorch Lightning model for the omni-neck dataset.
    It is used for training and evaluation of the model.
    """

    def __init__(
        self,
        x_dim: Tuple[int, ...],
        y_dim: Tuple[int, ...],
        h1_dim: Tuple[int, ...],
        h2_dim: Tuple[int, ...],
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
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        

        # Define the model architecture
        for i in range(len(self.y_dim)):
            setattr(
                self,
                f"estimator_{i}",
                nn.Sequential(
                    nn.Linear(self.x_dim[0], self.h1_dim[i]),
                    nn.ReLU(),
                    nn.Linear(self.h1_dim[i], self.h2_dim[i]),
                    nn.ReLU(),
                    nn.Linear(self.h2_dim[i], self.y_dim[i]),
                ),
            )

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
            x_hat_list: the predicted data.
        """
        
        x_hat_list = []
        for i in range(len(self.y_dim)):
            x_hat = getattr(self, f"estimator_{i}")(x)
            x_hat_list.append(x_hat)

        # Return the output data
        return x_hat_list

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

        x_hat = getattr(self, f"estimator_{output_index}")(x)

        # Return the output data
        return x_hat

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

        # Run the step
        y_hat_list = []

        for i in range(len(self.y_dim)):
            y_hat = getattr(self, f"estimator_{i}")(data_list[0])
            y_hat_list.append(y_hat)

        # Define the logs
        logs = {}

        # Calculate the reconstruction loss
        loss_list = []
        for i in range(len(self.y_dim)):
            y_i = data_list[i + 1]
            y_hat_i = y_hat_list[i]
            loss = F.mse_loss(y_hat_i, y_i)
            loss_list.append(loss)
            logs[f"loss_{i}"] = loss

        # Calculate the total loss
        loss = sum(loss_list)

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
            params += list(getattr(self, f"estimator_{i}").parameters())
        
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
