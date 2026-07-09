"""Guarded DQN-style value network utilities for training builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.training import torch_backend
from src.training.contracts import ACTION_OUTPUT_SIZE, OBSERVATION_INPUT_SIZE


@dataclass(frozen=True)
class ValueNetworkConfig:
    hidden_layer_width: int
    hidden_layer_count: int
    input_size: int = OBSERVATION_INPUT_SIZE
    output_count: int = ACTION_OUTPUT_SIZE
    optimizer: str = "adam"
    loss: str = "huber"

    def validate(self) -> None:
        if self.input_size != OBSERVATION_INPUT_SIZE:
            raise ValueError(f"input_size must be {OBSERVATION_INPUT_SIZE}")
        if self.output_count != ACTION_OUTPUT_SIZE:
            raise ValueError(f"output_count must be {ACTION_OUTPUT_SIZE}")
        if self.hidden_layer_width <= 0:
            raise ValueError("hidden_layer_width must be positive")
        if self.hidden_layer_count <= 0:
            raise ValueError("hidden_layer_count must be positive")
        if self.optimizer != "adam":
            raise ValueError("only adam optimizer is currently supported")
        if self.loss != "huber":
            raise ValueError("only huber loss is currently supported")


def build_value_network(config: ValueNetworkConfig, device: Any | None = None):
    """Build a fully connected ReLU network with unrestricted linear outputs."""
    config.validate()
    torch = torch_backend.require_torch()
    layers = []
    input_size = config.input_size
    for _ in range(config.hidden_layer_count):
        layers.append(torch.nn.Linear(input_size, config.hidden_layer_width))
        layers.append(torch.nn.ReLU())
        input_size = config.hidden_layer_width
    layers.append(torch.nn.Linear(input_size, config.output_count))
    model = torch.nn.Sequential(*layers)
    if device is not None:
        model = model.to(device)
    return model


def build_optimizer(model, learning_rate: float, name: str = "adam"):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if name != "adam":
        raise ValueError("only adam optimizer is currently supported")
    torch = torch_backend.require_torch()
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def _as_observation_tensor(observations, torch, device=None):
    tensor = observations
    if not hasattr(tensor, "shape"):
        tensor = torch.tensor(tensor, dtype=torch.float32, device=device)
    else:
        tensor = tensor.to(device=device, dtype=torch.float32) if device else tensor.float()
    if tensor.ndim != 2 or tensor.shape[1] != OBSERVATION_INPUT_SIZE:
        raise ValueError(f"observations must have shape [batch, {OBSERVATION_INPUT_SIZE}]")
    return tensor


def predict_action_values(model, observations):
    """Return raw action values for a batch of observations."""
    torch = torch_backend.require_torch()
    device = next(model.parameters()).device
    tensor = _as_observation_tensor(observations, torch, device)
    model.eval()
    with torch.no_grad():
        return model(tensor)


def selected_action_regression_loss(model, observations, action_indices, returns):
    """Compute Huber loss for only the selected action output per sample."""
    torch = torch_backend.require_torch()
    device = next(model.parameters()).device
    observation_tensor = _as_observation_tensor(observations, torch, device)
    action_tensor = torch.as_tensor(action_indices, dtype=torch.long, device=device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if action_tensor.ndim != 1:
        raise ValueError("action_indices must be a 1D batch")
    if return_tensor.ndim != 1:
        raise ValueError("returns must be a 1D batch")
    if observation_tensor.shape[0] != action_tensor.shape[0]:
        raise ValueError("observation and action batches must be the same size")
    if observation_tensor.shape[0] != return_tensor.shape[0]:
        raise ValueError("observation and return batches must be the same size")
    predictions = model(observation_tensor)
    selected_predictions = predictions.gather(1, action_tensor.view(-1, 1)).squeeze(1)
    return torch.nn.functional.smooth_l1_loss(selected_predictions, return_tensor)


def train_selected_action_regression(
    model,
    optimizer,
    observations,
    action_indices,
    returns,
) -> float:
    model.train()
    optimizer.zero_grad()
    loss = selected_action_regression_loss(
        model,
        observations,
        action_indices,
        returns,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
