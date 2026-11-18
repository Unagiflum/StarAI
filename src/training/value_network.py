"""Guarded DQN-style value network utilities for training builds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.training import torch_backend
from src.training.contracts import (
    ACTION_OUTPUT_SIZE,
    DEFAULT_NEW_MODEL_VALUE_HEAD,
    OBSERVATION_INPUT_SIZE,
    TRAINING_LOSS,
    VALUE_HEAD_DUELING,
    VALUE_HEAD_STANDARD,
    VALUE_HEADS,
)
from src.training.reflection import reflect_action_indices, reflect_observations


SUPPORTED_LOSSES = ("mse", "huber")
MAX_GRADIENT_NORM = 10.0


@dataclass(frozen=True)
class ValueNetworkConfig:
    hidden_layer_width: int
    hidden_layer_count: int
    input_size: int = OBSERVATION_INPUT_SIZE
    output_count: int = ACTION_OUTPUT_SIZE
    optimizer: str = "adam"
    loss: str = TRAINING_LOSS
    head: str = DEFAULT_NEW_MODEL_VALUE_HEAD

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
        if self.loss not in SUPPORTED_LOSSES:
            raise ValueError(f"loss must be one of {SUPPORTED_LOSSES}")
        if self.head not in VALUE_HEADS:
            raise ValueError(f"head must be one of {VALUE_HEADS}")


_DUELING_NETWORK_CLASS = None


def dueling_value_network_class():
    """Return the dueling module class, defined lazily so torch stays optional."""

    global _DUELING_NETWORK_CLASS
    if _DUELING_NETWORK_CLASS is None:
        torch = torch_backend.require_torch()

        class DuelingValueNetwork(torch.nn.Module):
            """Q(s, a) = V(s) + A(s, a) - mean_a A(s, a).

            Every replay sample trains the shared state value, while each
            sample's selected action only has to explain its advantage.
            """

            def __init__(self, trunk, value_head, advantage_head):
                super().__init__()
                self.trunk = trunk
                self.value_head = value_head
                self.advantage_head = advantage_head

            def forward(self, observations):
                features = self.trunk(observations)
                value = self.value_head(features)
                advantage = self.advantage_head(features)
                return value + advantage - advantage.mean(dim=-1, keepdim=True)

        _DUELING_NETWORK_CLASS = DuelingValueNetwork
    return _DUELING_NETWORK_CLASS


def is_dueling_value_network(model) -> bool:
    return (
        _DUELING_NETWORK_CLASS is not None
        and isinstance(model, _DUELING_NETWORK_CLASS)
    )


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
    if config.head == VALUE_HEAD_DUELING:
        model = dueling_value_network_class()(
            torch.nn.Sequential(*layers),
            torch.nn.Linear(input_size, 1),
            torch.nn.Linear(input_size, config.output_count),
        )
    else:
        layers.append(torch.nn.Linear(input_size, config.output_count))
        model = torch.nn.Sequential(*layers)
    if device is not None:
        model = model.to(device)
    return model


def value_network_config_from_architecture(
    architecture: Mapping[str, Any],
) -> ValueNetworkConfig:
    """Build a network config from normalized architecture metadata."""

    return ValueNetworkConfig(
        hidden_layer_width=int(architecture["hidden_layer_width"]),
        hidden_layer_count=int(architecture["hidden_layer_count"]),
        head=str(architecture.get("head", VALUE_HEAD_STANDARD)),
    )


def build_optimizer(model, learning_rate: float, name: str = "adam"):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if name != "adam":
        raise ValueError("only adam optimizer is currently supported")
    torch = torch_backend.require_torch()
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def _as_observation_tensor(observations, torch, device=None):
    tensor = observations
    if not torch.is_tensor(tensor):
        # Stack sequences of rows (including tuples of NumPy arrays) once;
        # converting a list of arrays directly is very slow in PyTorch.
        tensor = torch.as_tensor(
            np.asarray(tensor, dtype=np.float32),
            device=device,
        )
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


def predict_action_values_read_only(model, observations):
    """Return raw action values without mutating model mode."""
    torch = torch_backend.require_torch()
    device = next(model.parameters()).device
    tensor = _as_observation_tensor(observations, torch, device)
    inference_context = getattr(torch, "inference_mode", torch.no_grad)
    with inference_context():
        return model(tensor)


def selected_action_regression_loss(model, observations, action_indices, returns):
    """Compute mean-squared error for only the selected action output.

    Squared error estimates the mean return. Huber loss with unit delta behaves
    like absolute error for large returns and pulls rare large rewards, such as
    kills, toward the median.
    """
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
    return torch.nn.functional.mse_loss(selected_predictions, return_tensor)


def train_selected_action_regression(
    model,
    optimizer,
    observations,
    action_indices,
    returns,
) -> float:
    mirrored_observations = reflect_observations(observations)
    mirrored_action_indices = reflect_action_indices(action_indices)
    model.train()
    optimizer.zero_grad()
    original_loss = selected_action_regression_loss(
        model,
        observations,
        action_indices,
        returns,
    )
    (original_loss * 0.5).backward()
    mirrored_loss = selected_action_regression_loss(
        model,
        mirrored_observations,
        mirrored_action_indices,
        returns,
    )
    (mirrored_loss * 0.5).backward()
    clip_value_network_gradients(model)
    optimizer.step()
    combined_loss = 0.5 * (
        original_loss.detach() + mirrored_loss.detach()
    )
    return float(combined_loss.cpu().item())


def clip_value_network_gradients(model, max_norm: float = MAX_GRADIENT_NORM) -> None:
    """Bound one update's gradient norm; squared error grows with large misses."""

    torch = torch_backend.require_torch()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
