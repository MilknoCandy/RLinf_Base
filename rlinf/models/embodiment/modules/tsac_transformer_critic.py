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

"""Causal Transformer critic adapted from T-SAC for RLT Stage 2.

The critic consumes a state token ``z`` and a sequence of per-step actions, and
emits one prefix-conditioned Q-value per action position. Twin critics are two
independent copies with no shared weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TSACTransformerQHead(nn.Module):
    """Single causal Transformer Q-network.

    Sequence layout: ``[z, a_0, ..., a_{T-1}]``. Outputs are produced only at
    action positions, so ``forward`` returns ``[B, T, 1]``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_action_len: int = 25,
        ffn_mult: int = 4,
    ):
        super().__init__()
        state_dim = int(state_dim)
        action_dim = int(action_dim)
        d_model = int(d_model)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        max_action_len = int(max_action_len)
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if max_action_len < 1:
            raise ValueError(f"max_action_len must be >= 1, got {max_action_len}.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_action_len = max_action_len

        # T-SAC: separate state/action embeddings without bias.
        self.state_embed = nn.Linear(state_dim, d_model, bias=False)
        self.action_embed = nn.Linear(action_dim, d_model, bias=False)
        self.pos_embed = nn.Embedding(max_action_len + 1, d_model)

        encoder_layer_kwargs = {
            "d_model": d_model,
            "nhead": num_heads,
            "dim_feedforward": d_model * int(ffn_mult),
            "dropout": 0.0,
            "activation": "gelu",
            "batch_first": True,
            "norm_first": True,
        }
        try:
            encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_kwargs)
        except TypeError:
            # Older PyTorch: no batch_first / norm_first.
            encoder_layer_kwargs.pop("batch_first", None)
            encoder_layer_kwargs.pop("norm_first", None)
            encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_kwargs)
            self._encoder_batch_first = False
        else:
            self._encoder_batch_first = True
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        self.out_proj = nn.Linear(d_model, 1, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (self.state_embed, self.action_embed, self.out_proj):
            nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True means "ignore" for nn.TransformerEncoder when using src_mask as float.
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute prefix Q-values.

        Args:
            state: ``[B, state_dim]`` RL token (or state features).
            actions: ``[B, T, action_dim]`` realized action prefix.

        Returns:
            ``[B, T, 1]`` Q-value for every action prefix length ``1..T``.
        """
        if state.dim() != 2:
            raise ValueError(f"state must be [B, D], got {tuple(state.shape)}.")
        if actions.dim() != 3:
            raise ValueError(
                f"actions must be [B, T, A], got {tuple(actions.shape)}."
            )
        if state.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Batch mismatch: state {tuple(state.shape)} vs actions "
                f"{tuple(actions.shape)}."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_dim mismatch: expected {self.action_dim}, got "
                f"{actions.shape[-1]}."
            )
        seq_actions = int(actions.shape[1])
        if seq_actions < 1:
            raise ValueError("actions sequence length must be >= 1.")
        if seq_actions > self.max_action_len:
            raise ValueError(
                f"actions length {seq_actions} exceeds max_action_len "
                f"{self.max_action_len}."
            )

        state_tok = self.state_embed(state).unsqueeze(1)
        action_tok = self.action_embed(actions)
        tokens = torch.cat([state_tok, action_tok], dim=1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.pos_embed(positions)[None, :, :]

        causal_mask = self._causal_mask(tokens.shape[1], tokens.device)
        if getattr(self, "_encoder_batch_first", True):
            encoded = self.encoder(tokens, mask=causal_mask)
        else:
            # Legacy API expects [T, B, D].
            encoded = self.encoder(tokens.transpose(0, 1), mask=causal_mask)
            encoded = encoded.transpose(0, 1)
        q_values = self.out_proj(encoded[:, 1:, :])
        return q_values


class MultiTSACTransformerQHead(nn.Module):
    """Twin causal Transformer critics; last dim is the Q-head index."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_action_len: int = 25,
        num_q_heads: int = 2,
        ffn_mult: int = 4,
    ):
        super().__init__()
        num_q_heads = int(num_q_heads)
        if num_q_heads < 1:
            raise ValueError(f"num_q_heads must be >= 1, got {num_q_heads}.")
        self.num_q_heads = num_q_heads
        self.max_action_len = int(max_action_len)
        self.qs = nn.ModuleList(
            [
                TSACTransformerQHead(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    max_action_len=max_action_len,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_q_heads)
            ]
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return twin prefix Q-values with shape ``[B, T, num_q_heads]``."""
        q_vs = [qf(state, actions) for qf in self.qs]
        return torch.cat(q_vs, dim=-1)

    def q_id_forward(
        self, q_id: int, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.qs[q_id](state, actions)


def sample_tsac_horizon(
    horizons: list[int] | tuple[int, ...],
    valid_steps: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one horizon per row from ``horizons`` capped by ``valid_steps``.

    Args:
        horizons: Candidate control-step horizons, e.g. ``(5, 10, 15, 20, 25)``.
        valid_steps: ``[B]`` maximum realized prefix length available.
        generator: Optional RNG on the same device as ``valid_steps``.

    Returns:
        ``[B]`` long tensor of sampled horizons (at least the smallest feasible).
    """
    if not horizons:
        raise ValueError("horizons must be non-empty.")
    sorted_h = sorted(int(h) for h in horizons)
    if any(h < 1 for h in sorted_h):
        raise ValueError(f"horizons must be positive, got {sorted_h}.")
    valid = valid_steps.reshape(-1).to(dtype=torch.long)
    batch = int(valid.shape[0])
    device = valid.device
    sampled = torch.empty(batch, dtype=torch.long, device=device)
    horizon_tensor = torch.tensor(sorted_h, dtype=torch.long, device=device)
    min_h = int(sorted_h[0])
    for idx in range(batch):
        available = int(valid[idx].item())
        if available < min_h:
            sampled[idx] = max(available, 1)
            continue
        candidates = horizon_tensor[horizon_tensor <= available]
        # Uniform over feasible horizons.
        choice = torch.randint(
            low=0,
            high=int(candidates.numel()),
            size=(1,),
            generator=generator,
            device=device,
        )
        sampled[idx] = candidates[choice]
    return sampled


def prefix_horizon_mask(
    sampled_horizons: torch.Tensor,
    horizons: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Boolean mask ``[B, H]`` where entry ``(b, i)`` is active if ``horizons[i] <= n_b``."""
    horizon_tensor = torch.tensor(
        sorted(int(h) for h in horizons),
        dtype=torch.long,
        device=sampled_horizons.device,
    )
    return horizon_tensor[None, :] <= sampled_horizons.reshape(-1, 1)


def discounted_prefix_returns(
    rewards: torch.Tensor,
    gamma: float,
    horizons: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Cumulative discounted rewards for each horizon.

    Args:
        rewards: ``[B, T]`` per-control-step rewards.
        gamma: Discount factor.
        horizons: Horizon lengths to materialize.

    Returns:
        ``[B, H]`` discounted sums ``sum_{j=0}^{h-1} gamma^j r_j`` for each ``h``.
    """
    rewards = rewards.to(dtype=torch.float32)
    batch, total_steps = rewards.shape
    discounts = torch.pow(
        torch.as_tensor(gamma, dtype=rewards.dtype, device=rewards.device),
        torch.arange(total_steps, dtype=rewards.dtype, device=rewards.device),
    )
    discounted = rewards * discounts[None, :]
    cumsum = torch.cumsum(discounted, dim=-1)
    outs = []
    for horizon in sorted(int(h) for h in horizons):
        if horizon < 1 or horizon > total_steps:
            raise ValueError(
                f"horizon {horizon} out of range for rewards length {total_steps}."
            )
        outs.append(cumsum[:, horizon - 1])
    return torch.stack(outs, dim=-1)
