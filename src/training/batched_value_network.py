"""Batched execution helpers for same-shaped value networks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.training import torch_backend
from src.training.contracts import ACTION_OUTPUT_SIZE, OBSERVATION_INPUT_SIZE
from src.training.reflection import reflect_action_indices, reflect_observations


@dataclass(frozen=True)
class BatchedValueNetworkParameters:
    models: tuple[Any, ...]
    params: tuple[tuple[Any, Any], ...]
    device: Any
    dtype: Any

    @property
    def model_count(self) -> int:
        return len(self.models)


class BatchedValueNetworkParameterCache:
    """Bounded LRU cache of stacked parameters for stable model tuples."""

    def __init__(self, max_entries: int = 8):
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[
            tuple[int, ...],
            BatchedValueNetworkParameters,
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def get(
        self,
        models: Sequence[Any],
        *,
        set_eval: bool = False,
    ) -> BatchedValueNetworkParameters:
        model_tuple = tuple(models)
        key = tuple(id(model) for model in model_tuple)
        cached = self._entries.get(key)
        if cached is not None and all(
            cached_model is model
            for cached_model, model in zip(cached.models, model_tuple)
        ):
            self._entries.move_to_end(key)
            return cached
        cached = build_batched_value_network_parameters(
            model_tuple,
            set_eval=set_eval,
        )
        self._entries[key] = cached
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return cached


def can_batch_value_networks(models: Sequence[Any]) -> bool:
    """Return whether models can share the manual batched value-network path."""

    torch = torch_backend.get_torch()
    if torch is None or not models:
        return False
    try:
        _stack_linear_parameters(models, torch)
    except (TypeError, ValueError, StopIteration):
        return False
    return True


def build_batched_value_network_parameters(
    models: Sequence[Any],
    *,
    set_eval: bool = False,
) -> BatchedValueNetworkParameters:
    """Stack model parameters once for repeated inference calls."""

    torch = torch_backend.require_torch()
    model_tuple = tuple(models)
    if not model_tuple:
        raise ValueError("at least one model is required")
    params = _stack_linear_parameters(model_tuple, torch, detach=True)
    if set_eval:
        for model in model_tuple:
            model.eval()
    return BatchedValueNetworkParameters(
        models=model_tuple,
        params=params,
        device=params[0][0].device,
        dtype=params[0][0].dtype,
    )


def predict_action_values_from_batched_parameters(
    parameters: BatchedValueNetworkParameters,
    observations: Sequence[Sequence[float]],
):
    """Return action values using a previously stacked inference parameter bundle."""

    torch = torch_backend.require_torch()
    numeric_observations = np.asarray(observations, dtype=np.float32)
    tensor = torch.as_tensor(
        numeric_observations,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    expected_shape = (parameters.model_count, OBSERVATION_INPUT_SIZE)
    if tensor.ndim != 2 or tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"observations must have shape [model_count, {OBSERVATION_INPUT_SIZE}]"
        )
    inference_context = getattr(torch, "inference_mode", torch.no_grad)
    with inference_context():
        return _batched_forward(parameters.params, tensor)


def predict_action_values_batched(
    models: Sequence[Any],
    observations: Sequence[Sequence[float]],
    *,
    set_eval: bool = False,
    parameter_cache: BatchedValueNetworkParameterCache | None = None,
):
    """Return one action-value row per model/input pair."""

    torch = torch_backend.require_torch()
    if len(models) != len(observations):
        raise ValueError("models and observations must have the same length")
    if not models:
        return torch.empty((0, ACTION_OUTPUT_SIZE), dtype=torch.float32)
    parameters = (
        parameter_cache.get(models, set_eval=set_eval)
        if parameter_cache is not None
        else build_batched_value_network_parameters(models, set_eval=set_eval)
    )
    return predict_action_values_from_batched_parameters(parameters, observations)


def train_selected_action_regression_batched(
    models: Sequence[Any],
    optimizers: Sequence[Any],
    observations_by_model: Sequence[Sequence[Sequence[float]]],
    action_indices_by_model: Sequence[Sequence[int]],
    returns_by_model: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Train same-shaped models on original and reflected replay minibatches."""

    torch = torch_backend.require_torch()
    model_count = len(models)
    if not (
        model_count
        == len(optimizers)
        == len(observations_by_model)
        == len(action_indices_by_model)
        == len(returns_by_model)
    ):
        raise ValueError("batched training inputs must have the same model count")
    if model_count == 0:
        return ()

    params = _stack_linear_parameters(models, torch)
    device = params[0][0].device
    dtype = params[0][0].dtype
    observations = torch.as_tensor(
        _stack_numpy_batches(observations_by_model),
        dtype=dtype,
        device=device,
    )
    actions = torch.as_tensor(
        _stack_numpy_batches(action_indices_by_model),
        dtype=torch.long,
        device=device,
    )
    returns = torch.as_tensor(
        _stack_numpy_batches(returns_by_model),
        dtype=dtype,
        device=device,
    )
    if observations.ndim != 3 or observations.shape[0] != model_count:
        raise ValueError("observations must have shape [model_count, batch, input]")
    if observations.shape[2] != OBSERVATION_INPUT_SIZE:
        raise ValueError(f"observations must end with {OBSERVATION_INPUT_SIZE} values")
    if actions.shape != observations.shape[:2]:
        raise ValueError("action_indices must have shape [model_count, batch]")
    if returns.shape != observations.shape[:2]:
        raise ValueError("returns must have shape [model_count, batch]")

    for model, optimizer in zip(models, optimizers):
        model.train()
        optimizer.zero_grad()

    predictions = _batched_forward(params, observations)
    selected = predictions.gather(2, actions.unsqueeze(2)).squeeze(2)
    per_sample_losses = torch.nn.functional.smooth_l1_loss(
        selected,
        returns,
        reduction="none",
    )
    per_model_losses = per_sample_losses.mean(dim=1)
    (per_model_losses.sum() * 0.5).backward()
    original_losses = per_model_losses.detach()

    mirrored_observations = reflect_observations(observations)
    mirrored_actions = reflect_action_indices(actions)
    del observations, actions, predictions, selected, per_sample_losses
    del per_model_losses, params

    # Rebuild the stacked parameter graph so the reflected half can release the
    # original activations before accumulating its equally weighted gradients.
    params = _stack_linear_parameters(models, torch)
    predictions = _batched_forward(params, mirrored_observations)
    selected = predictions.gather(2, mirrored_actions.unsqueeze(2)).squeeze(2)
    per_sample_losses = torch.nn.functional.smooth_l1_loss(
        selected,
        returns,
        reduction="none",
    )
    per_model_losses = per_sample_losses.mean(dim=1)
    (per_model_losses.sum() * 0.5).backward()
    combined_losses = 0.5 * (original_losses + per_model_losses.detach())
    detached_losses = tuple(
        float(loss)
        for loss in combined_losses.cpu().tolist()
    )
    for optimizer in optimizers:
        optimizer.step()
    return detached_losses


def _batched_forward(params, observations):
    squeeze_model_batch = observations.ndim == 2
    values = observations.unsqueeze(1) if squeeze_model_batch else observations
    for layer_index, (weight, bias) in enumerate(params):
        values = values.bmm(weight.transpose(1, 2)) + bias.unsqueeze(1)
        if layer_index < len(params) - 1:
            values = values.relu()
    return values.squeeze(1) if squeeze_model_batch else values


def _stack_numpy_batches(values):
    if isinstance(values, np.ndarray):
        return values
    if len(values) > 0 and all(isinstance(value, np.ndarray) for value in values):
        return np.stack(values, axis=0)
    return values


def _stack_linear_parameters(models: Sequence[Any], torch, *, detach: bool = False):
    layer_groups = [_linear_layers_for_model(model, torch) for model in models]
    reference = layer_groups[0]
    reference_shapes = tuple(
        (
            tuple(layer.weight.shape),
            tuple(layer.bias.shape),
            layer.weight.device,
            layer.weight.dtype,
        )
        for layer in reference
    )
    params = []
    for layer_index, _layer in enumerate(reference):
        weights = []
        biases = []
        for layers in layer_groups:
            layer = layers[layer_index]
            layer_shape = (
                tuple(layer.weight.shape),
                tuple(layer.bias.shape),
                layer.weight.device,
                layer.weight.dtype,
            )
            if layer_shape != reference_shapes[layer_index]:
                raise ValueError(
                    "value networks must have matching shapes, devices, and dtypes"
                )
            weights.append(layer.weight.detach() if detach else layer.weight)
            biases.append(layer.bias.detach() if detach else layer.bias)
        params.append((torch.stack(weights, dim=0), torch.stack(biases, dim=0)))
    return tuple(params)


def _linear_layers_for_model(model: Any, torch):
    if not isinstance(model, torch.nn.Sequential):
        raise TypeError("batched value-network execution requires nn.Sequential models")
    modules = list(model)
    if len(modules) < 3 or len(modules) % 2 == 0:
        raise TypeError("unsupported value-network module layout")
    for index, module in enumerate(modules):
        if index % 2 == 0:
            if not isinstance(module, torch.nn.Linear):
                raise TypeError("expected Linear value-network layers")
        elif not isinstance(module, torch.nn.ReLU):
            raise TypeError("expected ReLU hidden activations")
    if modules[-1].out_features != ACTION_OUTPUT_SIZE:
        raise ValueError("value-network output size is incompatible")
    if modules[0].in_features != OBSERVATION_INPUT_SIZE:
        raise ValueError("value-network input size is incompatible")
    return tuple(module for module in modules if isinstance(module, torch.nn.Linear))
