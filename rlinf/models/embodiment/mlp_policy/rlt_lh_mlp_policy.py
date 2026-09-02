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

"""Stage-2 long-horizon RLT MLP actor-critic policy.

Compared with :class:`RLTMLPPolicy`, this policy adds a small temporal history
encoder so actor and critic are conditioned on a sliding window of previous
chunk summaries. The rest of the actor-critic interface stays compatible with
the existing RLT AC training worker.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead
from rlinf.models.embodiment.modules.rlt_history_encoder import RLTHistoryEncoder
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init

__all__ = ["RLTLHMLPPolicy"]


class RLTLHMLPPolicy(RLTMLPPolicy):
    """Long-horizon RLT policy with a learnable history encoder.

    ``history`` observation key accepts either a raw window with shape
    ``[batch, window, history_input_dim]`` (recommended for replay) or a
    pre-encoded vector with shape ``[batch, history_hidden_dim]``.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        fixed_std: float = 0.002,
        history_input_dim: int | None = None,
        history_hidden_dim: int = 256,
        history_num_layers: int = 1,
    ) -> None:
        z_dim = int(z_dim)
        proprio_dim = int(proprio_dim)
        step_action_dim = int(action_dim)
        chunk_len = int(num_action_chunks)
        flat_action_dim = chunk_len * step_action_dim

        self.history_input_dim = int(
            history_input_dim
            if history_input_dim is not None
            else z_dim + proprio_dim + flat_action_dim + 3
        )
        self.history_hidden_dim = int(history_hidden_dim)
        self.history_num_layers = int(history_num_layers)

        super().__init__(
            z_dim=z_dim,
            proprio_dim=proprio_dim,
            action_dim=step_action_dim,
            num_action_chunks=chunk_len,
            ref_num_action_chunks=ref_num_action_chunks,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
            fixed_std=fixed_std,
        )

        actor_obs_dim = z_dim + proprio_dim + flat_action_dim + self.history_hidden_dim
        critic_obs_dim = z_dim + proprio_dim + self.history_hidden_dim
        self.obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim

        activation = "tanh"
        act = get_act_func(activation)
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(actor_obs_dim, 256)),
            act(),
            layer_init(nn.Linear(256, 256)),
            act(),
            layer_init(nn.Linear(256, 256)),
            act(),
        )
        self.actor_mean = layer_init(
            nn.Linear(256, flat_action_dim), std=0.01 * np.sqrt(2)
        )

        if add_q_head:
            if q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=critic_obs_dim,
                    hidden_dims=[256, 256, 256],
                    num_q_heads=2,
                    output_dim=self.num_action_chunks,
                    action_feature_dim=flat_action_dim,
                )
            elif q_head_type == "crossq":
                self.q_head = MultiCrossQHead(
                    hidden_size=critic_obs_dim,
                    hidden_dims=[256, 256, 256],
                    num_q_heads=2,
                    output_dim=self.num_action_chunks,
                    action_feature_dim=flat_action_dim,
                )
            else:
                raise ValueError(f"Invalid q_head_type: {q_head_type}")

        self.history_gru = RLTHistoryEncoder(
            input_dim=self.history_input_dim,
            hidden_dim=self.history_hidden_dim,
            num_layers=self.history_num_layers,
        )

    def _history_feature(self, obs: dict) -> torch.Tensor:
        history = obs["history"]
        if history.dim() == 3:
            return self.history_gru(history)
        if history.dim() == 2:
            if history.shape[-1] != self.history_hidden_dim:
                raise ValueError(
                    "Pre-encoded history dimension mismatch: expected "
                    f"{self.history_hidden_dim}, got {history.shape[-1]}."
                )
            return history
        raise ValueError(
            "history observation must have shape [batch, window, summary_dim] "
            f"or [batch, history_hidden_dim], got {tuple(history.shape)}."
        )

    def _actor_state(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        ref_chunk = self._get_ref_chunk(obs)
        if apply_reference_dropout:
            ref_chunk = self._maybe_drop_reference(ref_chunk, reference_dropout_prob)
        history_feature = self._history_feature(obs)
        return torch.cat(
            [ref_chunk, self._get_z(obs), self._get_proprio(obs), history_feature],
            dim=-1,
        )

    def _critic_state(self, obs: dict) -> torch.Tensor:
        history_feature = self._history_feature(obs)
        return torch.cat(
            [self._get_z(obs), self._get_proprio(obs), history_feature],
            dim=-1,
        )
