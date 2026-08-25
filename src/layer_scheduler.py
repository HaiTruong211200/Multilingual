"""POTSA-style reward-guided selection of contrastive alignment layers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional

import torch


@dataclass
class LayerState:
    q_value: float = 0.0
    count: int = 0
    prev_task_loss: Optional[float] = None


class POTSAStyleLayerScheduler:
    """Select layers using loss-change reward, EMA, UCB, and softmax."""

    def __init__(
        self,
        candidate_layers: Iterable[int],
        ema_rho: float = 0.1,
        ucb_beta: float = 0.5,
        temperature: float = 1.0,
        warmup_steps: int = 0,
        force_each_layer_once: bool = True,
        seed: int = 42,
    ) -> None:
        self.candidate_layers = tuple(candidate_layers)
        if not self.candidate_layers:
            raise ValueError("candidate_layers cannot be empty")
        if len(set(self.candidate_layers)) != len(self.candidate_layers):
            raise ValueError("candidate_layers must not contain duplicates")
        if not 0.0 < ema_rho <= 1.0:
            raise ValueError("ema_rho must be in (0, 1]")
        if ucb_beta < 0.0:
            raise ValueError("ucb_beta cannot be negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")

        self.ema_rho = ema_rho
        self.ucb_beta = ucb_beta
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.force_each_layer_once = force_each_layer_once
        self.states = {layer: LayerState() for layer in self.candidate_layers}
        self.generator = torch.Generator().manual_seed(seed)
        self.current_layer = self.candidate_layers[0]

    def observe(self, layer_idx: int, task_loss: float) -> Optional[float]:
        """Update the selected layer and return its raw loss-change reward."""
        if layer_idx not in self.states:
            raise ValueError(f"Layer {layer_idx} is not a candidate")
        if not math.isfinite(task_loss):
            return None

        state = self.states[layer_idx]
        reward = None
        if state.prev_task_loss is not None:
            reward = state.prev_task_loss - task_loss
            state.q_value = (
                (1.0 - self.ema_rho) * state.q_value
                + self.ema_rho * reward
            )
        state.prev_task_loss = task_loss
        state.count += 1
        return reward

    def get_utilities(self, global_step: int) -> Dict[int, float]:
        step = max(global_step, 2)
        return {
            layer: state.q_value
            + self.ucb_beta
            * math.sqrt(math.log(step) / max(1, state.count))
            for layer, state in self.states.items()
        }

    def get_probabilities(self, global_step: int) -> Dict[int, float]:
        utilities = self.get_utilities(global_step)
        values = torch.tensor(
            [utilities[layer] for layer in self.candidate_layers],
            dtype=torch.float64,
        )
        probabilities = torch.softmax(values / self.temperature, dim=0)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Layer scheduler produced non-finite probabilities")
        return {
            layer: float(probabilities[index])
            for index, layer in enumerate(self.candidate_layers)
        }

    def sample_next(self, global_step: int) -> int:
        if self.force_each_layer_once:
            for layer in self.candidate_layers:
                if self.states[layer].count == 0:
                    self.current_layer = layer
                    return layer

        if global_step < self.warmup_steps:
            index = int(
                torch.randint(
                    len(self.candidate_layers), (1,), generator=self.generator
                ).item()
            )
        else:
            probabilities = self.get_probabilities(global_step)
            values = torch.tensor(
                [probabilities[layer] for layer in self.candidate_layers]
            )
            index = int(torch.multinomial(values, 1, generator=self.generator).item())
        self.current_layer = self.candidate_layers[index]
        return self.current_layer

    def state_dict(self) -> dict:
        return {
            "candidate_layers": self.candidate_layers,
            "ema_rho": self.ema_rho,
            "ucb_beta": self.ucb_beta,
            "temperature": self.temperature,
            "warmup_steps": self.warmup_steps,
            "force_each_layer_once": self.force_each_layer_once,
            "current_layer": self.current_layer,
            "states": {layer: asdict(state) for layer, state in self.states.items()},
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if tuple(state_dict["candidate_layers"]) != self.candidate_layers:
            raise ValueError("Checkpoint candidate layers do not match configuration")
        for layer, values in state_dict["states"].items():
            self.states[int(layer)] = LayerState(**values)
        self.current_layer = int(state_dict["current_layer"])
        self.generator.set_state(state_dict["generator_state"])
