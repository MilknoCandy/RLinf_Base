# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLT prefix-encoder loop used as memory.

``z_t = Encoder([I_t; a_{t-1}; r_{t-1}; z_{t-1}])``. ``ref_chunk`` is never an
encoder input; it is only a prediction target. Training unrolls a same-episode
window so ``z_t`` depends on recent ``(I, a, r)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder


def pool_rlt_prefix(
    prefix_embs: torch.Tensor,
    prefix_mask: torch.Tensor | None,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stride-pool prefix tokens so the loop can store and replay ``I_t``."""
    length = int(length)
    if prefix_embs.shape[1] == length:
        if prefix_mask is None:
            prefix_mask = torch.ones(
                prefix_embs.shape[0],
                length,
                device=prefix_embs.device,
                dtype=torch.bool,
            )
        return prefix_embs, prefix_mask.to(dtype=torch.bool)
    pooled = F.adaptive_avg_pool1d(prefix_embs.transpose(1, 2), length).transpose(1, 2)
    mask = torch.ones(
        prefix_embs.shape[0],
        length,
        device=prefix_embs.device,
        dtype=torch.bool,
    )
    return pooled, mask


class RLTLoopEncoder(nn.Module):
    """Trainable Stage-1-style encoder that loops ``z`` across chunks."""

    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        prefix_len: int,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        z_dim = int(z_dim)
        self.z_dim = z_dim
        self.action_dim = int(action_dim)
        self.prefix_len = int(prefix_len)
        self.encoder = RLTTokenEncoder(
            input_dim=z_dim,
            embed_dim=z_dim,
            prefix_seq_len=self.prefix_len,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
        )
        self.mem_init = nn.Parameter(torch.zeros(z_dim))
        self.action_proj = nn.Linear(self.action_dim, z_dim)
        self.reward_proj = nn.Linear(1, z_dim)
        self.pred_ref = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, self.action_dim),
        )
        self.pred_r = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, 1),
        )

    def initial_mem(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return self.mem_init.to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1
        )

    def extra_tokens(
        self, prev_action: torch.Tensor, prev_reward: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            [self.action_proj(prev_action), self.reward_proj(prev_reward)],
            dim=1,
        )

    def write(
        self,
        prefix_embs: torch.Tensor,
        prefix_mask: torch.Tensor | None,
        z_prev: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        *,
        detach_prev: bool = True,
    ) -> torch.Tensor:
        if detach_prev:
            z_prev = z_prev.detach()
        prefix_embs = prefix_embs.to(dtype=z_prev.dtype)
        encoded = self.encoder(
            prefix_embs,
            prefix_mask,
            rl_token=z_prev,
            extra_tokens=self.extra_tokens(prev_action, prev_reward),
        )
        return encoded.reshape(prefix_embs.shape[0], -1)

    def unroll(
        self,
        hist_prefix: torch.Tensor,
        hist_mask: torch.Tensor | None,
        hist_action: torch.Tensor,
        hist_reward: torch.Tensor,
        hist_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write \(z\) through a same-episode window. Gradients flow across steps."""
        if hist_prefix.dim() == 5:
            hist_prefix = hist_prefix.reshape(
                hist_prefix.shape[0], -1, hist_prefix.shape[-2], hist_prefix.shape[-1]
            )
        batch, steps, _, _ = hist_prefix.shape
        device, dtype = hist_prefix.device, hist_prefix.dtype
        valid = hist_valid.to(device=device).reshape(batch, steps, 1).to(dtype=dtype)
        z = self.initial_mem(batch, device, dtype)
        prev_action = torch.zeros(
            batch, self.action_dim, device=device, dtype=dtype
        )
        prev_reward = torch.zeros(batch, 1, device=device, dtype=dtype)
        z_seq = []
        for step in range(steps):
            mask_t = None if hist_mask is None else hist_mask[:, step]
            z_new = self.write(
                hist_prefix[:, step],
                mask_t,
                z,
                prev_action,
                prev_reward,
                detach_prev=False,
            )
            z = z_new * valid[:, step] + z * (1.0 - valid[:, step])
            action_t = hist_action[:, step].to(device=device, dtype=dtype)
            reward_t = hist_reward[:, step].to(device=device, dtype=dtype)
            if reward_t.dim() == 1:
                reward_t = reward_t.unsqueeze(-1)
            prev_action = action_t * valid[:, step] + prev_action * (
                1.0 - valid[:, step]
            )
            prev_reward = reward_t * valid[:, step] + prev_reward * (
                1.0 - valid[:, step]
            )
            z_seq.append(z)
        return z, torch.stack(z_seq, dim=1), hist_valid.to(device=device).reshape(
            batch, steps
        )

    def predict(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pred_ref(z), self.pred_r(z)
