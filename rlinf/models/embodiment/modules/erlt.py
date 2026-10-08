# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""eRLT routing token, layer router, and the LIBERO actor-critic heads.

The frozen VLA is not owned here. This module only holds the parameters that
Appendix B of eRLT trains: routing-token embeddings, task-level layer logits,
the temporary action probe, and the online latent-noise actor and Q ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

LIBERO_LAYER_INDICES: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18)


def is_erlt_routing_update(update_step: int, interval: int) -> bool:
    """Return whether this 0-based learner step updates the router.

    The paper fixes the router for the first 199 updates of every 200 and
    opens its gradient on the last one. ``update_step`` is the counter before
    it is incremented, so step 199, 399, ... are routing steps when
    ``interval`` is 200.
    """
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}.")
    if update_step < 0:
        raise ValueError(f"update_step must be >= 0, got {update_step}.")
    return (int(update_step) + 1) % int(interval) == 0


def expert_action_target(
    actions: torch.Tensor,
    *,
    action_chunk: int,
    action_dim: int,
) -> torch.Tensor:
    """Flatten the normalized expert chunk used by the stage-1 probe.

    Accepts ``[B, H, A]`` or ``[B, H * A]``. Only the first ``action_chunk``
    steps and ``action_dim`` environment dimensions are kept, matching the
    LIBERO target of ``10 x 7``.
    """
    if action_chunk <= 0 or action_dim <= 0:
        raise ValueError("action_chunk and action_dim must be positive.")
    if actions.dim() == 3:
        chunk = actions[:, :action_chunk, :action_dim]
        return chunk.reshape(actions.shape[0], -1)
    if actions.dim() == 2:
        width = action_chunk * action_dim
        if actions.shape[-1] < width:
            raise ValueError(
                f"flat actions need at least {width} dims, got {actions.shape[-1]}."
            )
        return actions[:, :width]
    raise ValueError(f"actions must be 2D or 3D, got {tuple(actions.shape)}.")


class ERLTRouter(nn.Module):
    """Learned routing tokens plus a task-level softmax over VLM depths."""

    def __init__(
        self,
        *,
        hidden_dim: int = 2048,
        num_tokens: int = 1,
        layer_indices: tuple[int, ...] = LIBERO_LAYER_INDICES,
        logit_slots: int = 64,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if logit_slots < len(layer_indices):
            raise ValueError(
                "logit_slots must cover every selected layer, got "
                f"{logit_slots} slots for {len(layer_indices)} layers."
            )
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.temperature = float(temperature)
        self.routing_tokens = nn.Parameter(
            torch.randn(self.num_tokens, self.hidden_dim) * 0.02
        )
        self.layer_logits = nn.Parameter(torch.zeros(int(logit_slots)))

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    def layer_weights(self) -> torch.Tensor:
        """Softmax over the active layer logits. Unused slots get no gradient."""
        logits = self.layer_logits[: self.num_layers]
        return torch.softmax(logits / self.temperature, dim=0)

    def combine(self, summaries: torch.Tensor) -> torch.Tensor:
        """Mix per-layer routing summaries into one RL token.

        Args:
            summaries: ``[B, M, D]`` mean-pooled routing states.

        Returns:
            ``[B, D]`` RL token.
        """
        if summaries.dim() != 3:
            raise ValueError(
                f"summaries must be [B, M, D], got {tuple(summaries.shape)}."
            )
        if summaries.shape[1] != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layer summaries, got {summaries.shape[1]}."
            )
        weights = self.layer_weights().to(dtype=summaries.dtype)
        return torch.einsum("m,bmd->bd", weights, summaries)


class ERLTActionProbe(nn.Module):
    """Temporary MLP discarded before online RL. Widths follow Appendix C.2."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, rl_token: torch.Tensor) -> torch.Tensor:
        return self.net(rl_token)


def _mlp_trunk(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(in_dim, width), nn.LayerNorm(width), nn.ReLU()])
        in_dim = width
    return nn.Sequential(*layers)


class ERLTLatentActor(nn.Module):
    """Gaussian latent-noise actor. One 32-d sample is repeated over the chunk.

    Log-std is initialized at 0 and clamped so the standard deviation stays in
    ``[exp(-20), exp(2)]``, as in the LIBERO eRLT setup. The sample is not
    squashed: it is the initial noise of the frozen flow policy.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
        action_horizon: int = 10,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
        self.output_dim = int(output_dim)
        self.action_horizon = int(action_horizon)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk = _mlp_trunk(input_dim, tuple(hidden_dims))
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.mean_layer = nn.Linear(last_dim, self.output_dim)
        self.log_std_layer = nn.Linear(last_dim, self.output_dim)
        nn.init.zeros_(self.log_std_layer.weight)
        nn.init.zeros_(self.log_std_layer.bias)

    def _distribution(self, rl_token: torch.Tensor) -> Independent:
        hidden = self.trunk(rl_token)
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden).clamp(self.log_std_min, self.log_std_max)
        base = Normal(mean, log_std.exp())
        return Independent(base, 1)

    def sample(
        self, rl_token: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[B, H, A]`` noise and a ``[B]`` log-probability."""
        dist = self._distribution(rl_token)
        latent = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(latent)
        noise = latent.unsqueeze(1).expand(-1, self.action_horizon, -1).contiguous()
        return noise, log_prob


class ERLTMultiQ(nn.Module):
    """Ten independent Q heads on the concatenated RL token and latent action."""

    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
        num_q_heads: int = 10,
    ) -> None:
        super().__init__()
        if num_q_heads <= 0:
            raise ValueError(f"num_q_heads must be positive, got {num_q_heads}.")
        self.action_dim = int(action_dim)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    _mlp_trunk(token_dim + action_dim, tuple(hidden_dims)),
                    nn.Linear(
                        hidden_dims[-1] if hidden_dims else token_dim + action_dim,
                        1,
                    ),
                )
                for _ in range(int(num_q_heads))
            ]
        )

    def forward(self, rl_token: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """Return ``[B, num_q_heads]``."""
        if latent_action.dim() == 3:
            latent_action = latent_action[:, 0, :]
        elif latent_action.dim() == 2 and latent_action.shape[-1] != self.action_dim:
            if latent_action.shape[-1] % self.action_dim != 0:
                raise ValueError(
                    "cannot reshape latent action "
                    f"{tuple(latent_action.shape)} into dim {self.action_dim}."
                )
            latent_action = latent_action[:, : self.action_dim]
        features = torch.cat([rl_token, latent_action], dim=-1)
        values = [head(features) for head in self.heads]
        return torch.cat(values, dim=-1)
