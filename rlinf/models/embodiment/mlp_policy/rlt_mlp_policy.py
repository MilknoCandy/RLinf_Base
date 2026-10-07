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

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.tsac_transformer_critic import (
    MultiTSACTransformerQHead,
)
from rlinf.models.embodiment.modules.value_head import ValueHead


class RLTMLPPolicy(MLPPolicy):
    """MLP actor-critic policy for RLT Stage 2 heads.

    Actor input follows RLT: reference action chunk, RL token feature, and
    proprioceptive state. Critic input is either the default MLP over
    ``(z_rl, proprio, action_chunk)`` or a T-SAC-style Transformer over
    ``(z_rl, per-step actions)``.
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
        mlp_hidden_dim: int = 256,
        mlp_num_hidden_layers: int = 3,
        frozen_action_dims: list[int] | tuple[int, ...] | None = None,
        tsac_d_model: int = 512,
        tsac_num_layers: int = 2,
        tsac_num_heads: int = 8,
        tsac_max_action_len: int = 25,
        action_extract: str = "sample",
        qc_num_samples: int = 8,
        qc_include_ref_chunk: bool = True,
        qc_include_actor_mean: bool = True,
        chunk_critic_steps: int | None = None,
        scale_critic_steps: list[int] | tuple[int, ...] | None = None,
        add_scale_value_heads: bool = False,
        aqc_gamma: float = 0.99,
    ):
        if not add_q_head:
            raise ValueError(
                "RLTMLPPolicy requires add_q_head=True for actor-critic training."
            )
        z_dim = int(z_dim)
        proprio_dim = int(proprio_dim)
        step_action_dim = int(action_dim)
        chunk_len = int(num_action_chunks)
        ref_chunk_len = (
            chunk_len if ref_num_action_chunks is None else int(ref_num_action_chunks)
        )
        if ref_chunk_len < chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{ref_chunk_len} < {chunk_len}."
            )
        flat_action_dim = chunk_len * step_action_dim
        self.q_head_type = str(q_head_type)
        self.use_tsac_critic = self.q_head_type == "tsac_transformer"

        actor_obs_dim = z_dim + proprio_dim + flat_action_dim
        critic_obs_dim = z_dim if self.use_tsac_critic else z_dim + proprio_dim

        parent_q_head_type = (
            "default" if self.use_tsac_critic else self.q_head_type
        )
        super().__init__(
            obs_dim=actor_obs_dim,
            action_dim=flat_action_dim,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=True,
            q_head_type=parent_q_head_type,
            critic_obs_dim=critic_obs_dim,
            hidden_dim=int(mlp_hidden_dim),
            num_hidden_layers=int(mlp_num_hidden_layers),
        )
        if self.use_tsac_critic:
            # Replace the temporary MLP twin-Q with the T-SAC Transformer.
            self.q_head = MultiTSACTransformerQHead(
                state_dim=z_dim,
                action_dim=step_action_dim,
                d_model=int(tsac_d_model),
                num_layers=int(tsac_num_layers),
                num_heads=int(tsac_num_heads),
                max_action_len=int(tsac_max_action_len),
                num_q_heads=2,
            )

        frozen_dims = tuple(int(dim) for dim in (frozen_action_dims or []))
        for dim in frozen_dims:
            if dim < 0 or dim >= step_action_dim:
                raise ValueError(
                    "frozen_action_dims must index the per-step action, got "
                    f"{dim} with action_dim={step_action_dim}."
                )
        self.frozen_action_dims = frozen_dims
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.step_action_dim = step_action_dim
        self.chunk_len = chunk_len
        self.ref_chunk_len = ref_chunk_len
        self.flat_action_dim = flat_action_dim
        self.fixed_std = float(fixed_std)
        if self.fixed_std <= 0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")

        extract = str(action_extract).lower()
        if extract not in {"sample", "best_of_n", "adaptive"}:
            raise ValueError(
                "action_extract must be sample, best_of_n, or adaptive, "
                f"got {action_extract!r}."
            )
        self.action_extract = extract
        self.qc_num_samples = int(qc_num_samples)
        self.qc_include_ref_chunk = bool(qc_include_ref_chunk)
        self.qc_include_actor_mean = bool(qc_include_actor_mean)
        self.aqc_gamma = float(aqc_gamma)
        hidden_dims = [int(mlp_hidden_dim)] * int(mlp_num_hidden_layers)

        self.chunk_critic_steps = (
            None if chunk_critic_steps is None else int(chunk_critic_steps)
        )
        if self.chunk_critic_steps is not None:
            if self.chunk_critic_steps < chunk_len:
                raise ValueError(
                    "chunk_critic_steps must be >= num_action_chunks, got "
                    f"{self.chunk_critic_steps} < {chunk_len}."
                )
            self.chunk_q_head = MultiQHead(
                hidden_size=critic_obs_dim,
                hidden_dims=hidden_dims,
                num_q_heads=2,
                output_dim=1,
                action_feature_dim=self.chunk_critic_steps * step_action_dim,
            )
            self.q_head_v = ValueHead(
                critic_obs_dim,
                hidden_sizes=tuple(hidden_dims),
                activation="tanh",
                output_dim=1,
            )

        raw_scales = [] if scale_critic_steps is None else list(scale_critic_steps)
        scales = sorted({int(step) for step in raw_scales})
        for step in scales:
            if step < 1 or step > chunk_len:
                raise ValueError(
                    "scale_critic_steps must be in [1, num_action_chunks], "
                    f"got {scales} with chunk_len={chunk_len}."
                )
        if chunk_len not in scales:
            scales.append(chunk_len)
            scales = sorted(set(scales))
        self.scale_critic_steps = tuple(scales)
        self.add_scale_value_heads = bool(add_scale_value_heads) or extract == "adaptive"
        if self.add_scale_value_heads or len(self.scale_critic_steps) > 1:
            self.scale_q_heads = torch.nn.ModuleDict()
            self.q_head_v_scale = torch.nn.ModuleDict()
            for step in self.scale_critic_steps:
                key = str(step)
                if step == chunk_len:
                    continue
                self.scale_q_heads[key] = MultiQHead(
                    hidden_size=critic_obs_dim,
                    hidden_dims=hidden_dims,
                    num_q_heads=2,
                    output_dim=1,
                    action_feature_dim=step * step_action_dim,
                )
                if self.add_scale_value_heads:
                    self.q_head_v_scale[key] = ValueHead(
                        critic_obs_dim,
                        hidden_sizes=tuple(hidden_dims),
                        activation="tanh",
                        output_dim=1,
                    )
            if self.add_scale_value_heads:
                self.q_head_v_scale[str(chunk_len)] = ValueHead(
                    critic_obs_dim,
                    hidden_sizes=tuple(hidden_dims),
                    activation="tanh",
                    output_dim=1,
                )
        else:
            self.scale_q_heads = torch.nn.ModuleDict()
            self.q_head_v_scale = torch.nn.ModuleDict()

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed = {}
        for key, value in env_obs.items():
            processed[key] = value.to(device) if torch.is_tensor(value) else value
        return processed

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _get_z(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["z_rl"])

    def _get_proprio(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["proprio"])

    def _get_ref_chunk(self, obs: dict) -> torch.Tensor:
        ref_chunk = self._flatten_batch(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, self.step_action_dim
        )
        ref_chunk = ref_chunk[:, : self.chunk_len]
        return ref_chunk.reshape(ref_chunk.shape[0], -1)

    def _maybe_drop_reference(
        self,
        ref_chunk: torch.Tensor,
        reference_dropout_prob: float,
    ) -> torch.Tensor:
        if reference_dropout_prob <= 0:
            return ref_chunk
        keep_prob = 1.0 - float(reference_dropout_prob)
        keep_mask = (
            torch.rand((ref_chunk.shape[0], 1), device=ref_chunk.device) < keep_prob
        )
        return ref_chunk * keep_mask.to(dtype=ref_chunk.dtype)

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
        return torch.cat([ref_chunk, self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _critic_state(self, obs: dict) -> torch.Tensor:
        if self.use_tsac_critic:
            return self._get_z(obs)
        return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def _hold_frozen_action_dims(
        self, actions: torch.Tensor, obs: dict
    ) -> torch.Tensor:
        """Keep selected per-step dims on the VLA proposal."""
        if not self.frozen_action_dims:
            return actions
        ref_chunk = self._get_ref_chunk(obs)
        if ref_chunk.shape != actions.shape:
            ref_chunk = ref_chunk.reshape_as(actions)
        held = actions.clone()
        step = self.step_action_dim
        for dim in self.frozen_action_dims:
            held[..., dim::step] = ref_chunk[..., dim::step]
        return held

    def _reshape_actions_for_tsac(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 3:
            return actions
        flat = self._flatten_batch(actions)
        if flat.shape[-1] % self.step_action_dim != 0:
            raise ValueError(
                "TSAC actions trailing dim must be divisible by action_dim, got "
                f"{flat.shape[-1]} and action_dim={self.step_action_dim}."
            )
        seq_len = flat.shape[-1] // self.step_action_dim
        return flat.reshape(flat.shape[0], seq_len, self.step_action_dim)

    def sac_forward(
        self,
        obs,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        deterministic: bool = False,
        **kwargs,
    ):
        actor_state = self._actor_state(
            obs,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        )
        feat = self.backbone(actor_state)
        action_mean = self.actor_mean(feat)
        action_std = torch.full_like(action_mean, self.fixed_std)
        probs = Normal(action_mean, action_std)
        action = action_mean if deterministic else probs.rsample()
        chunk_logprobs = probs.log_prob(action)
        action = torch.tanh(action)
        action = self._hold_frozen_action_dims(action, obs)
        return action, chunk_logprobs, None

    def _min_twin_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])

    def chunk_q_forward(self, obs, actions, detach_encoder: bool = False):
        if not hasattr(self, "chunk_q_head"):
            raise RuntimeError("chunk_q_head is not configured on this policy.")
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        flat = self._flatten_batch(actions)
        expected = int(self.chunk_critic_steps) * self.step_action_dim
        if flat.shape[-1] != expected:
            raise ValueError(
                "chunk critic action dim mismatch: expected "
                f"{expected}, got {tuple(flat.shape)}."
            )
        return self.chunk_q_head(critic_state, flat)

    def scale_q_forward(self, obs, actions, horizon: int, detach_encoder: bool = False):
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        prefix = self._flatten_batch(actions)
        expected = int(horizon) * self.step_action_dim
        if prefix.shape[-1] < expected:
            raise ValueError(
                f"scale critic {horizon} needs {expected} action dims, "
                f"got {tuple(prefix.shape)}."
            )
        prefix = prefix[..., :expected]
        if int(horizon) == self.chunk_len:
            return self.q_head(critic_state, prefix)
        key = str(int(horizon))
        if key not in self.scale_q_heads:
            raise KeyError(f"No scale Q head for horizon={horizon}.")
        return self.scale_q_heads[key](critic_state, prefix)

    def scale_v_forward(self, obs, horizon: int, detach_encoder: bool = False):
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        key = str(int(horizon))
        if key in self.q_head_v_scale:
            return self.q_head_v_scale[key](critic_state)
        if hasattr(self, "q_head_v"):
            return self.q_head_v(critic_state)
        raise KeyError(f"No value head for horizon={horizon}.")

    def _gather_extract_candidates(self, obs, *, num_samples: int) -> torch.Tensor:
        from rlinf.algorithms.qc.best_of_n import (
            flatten_chunk_actions,
            stack_action_candidates,
        )

        parts: list[torch.Tensor] = []
        if self.qc_include_ref_chunk:
            parts.append(
                flatten_chunk_actions(
                    self._get_ref_chunk(obs),
                    chunk_len=self.chunk_len,
                    action_dim=self.step_action_dim,
                )
            )
        mean_actions, _, _ = self.sac_forward(obs, deterministic=True)
        if self.qc_include_actor_mean:
            parts.append(
                flatten_chunk_actions(
                    mean_actions,
                    chunk_len=self.chunk_len,
                    action_dim=self.step_action_dim,
                )
            )
        if num_samples > 0:
            from rlinf.algorithms.qc.best_of_n import repeat_obs

            batch = int(mean_actions.shape[0])
            expanded = repeat_obs(obs, num_samples)
            sampled, _, _ = self.sac_forward(expanded, deterministic=False)
            flat = flatten_chunk_actions(
                sampled,
                chunk_len=self.chunk_len,
                action_dim=self.step_action_dim,
            )
            parts.append(flat.reshape(batch, num_samples, -1))
        if not parts:
            raise ValueError("Best-of-N extraction needs at least one candidate.")
        return stack_action_candidates(parts)

    def _best_of_n_actions(self, obs) -> tuple[torch.Tensor, dict[str, float]]:
        from rlinf.algorithms.qc.best_of_n import select_best_of_n_actions

        candidates = self._gather_extract_candidates(
            obs, num_samples=self.qc_num_samples
        )
        batch, num_candidates, _ = candidates.shape
        expanded = {}
        for key, value in obs.items():
            if not torch.is_tensor(value):
                expanded[key] = value
                continue
            expanded[key] = (
                value.unsqueeze(1)
                .expand(value.shape[0], num_candidates, *value.shape[1:])
                .reshape(batch * num_candidates, *value.shape[1:])
            )
        all_q = self.sac_q_forward(
            expanded, candidates.reshape(batch * num_candidates, -1)
        )
        q_values = self._min_twin_q(all_q).reshape(batch, num_candidates)
        best, indices = select_best_of_n_actions(q_values, candidates)
        metrics = {
            "qc/num_candidates": float(num_candidates),
            "qc/q_best_mean": q_values.max(dim=-1).values.mean().item(),
        }
        return best, metrics

    def _adaptive_actions(self, obs) -> tuple[torch.Tensor, dict[str, float]]:
        from rlinf.algorithms.qc.adaptive import (
            discount_normalized_advantage,
            select_adaptive_chunk,
            zscore,
        )

        candidates = self._gather_extract_candidates(
            obs, num_samples=self.qc_num_samples
        )
        batch, num_candidates, _ = candidates.shape
        horizons = self.scale_critic_steps
        scores = []
        with torch.no_grad():
            for horizon in horizons:
                prefix = candidates[..., : horizon * self.step_action_dim]
                expanded = {}
                for key, value in obs.items():
                    if not torch.is_tensor(value):
                        expanded[key] = value
                        continue
                    expanded[key] = (
                        value.unsqueeze(1)
                        .expand(value.shape[0], num_candidates, *value.shape[1:])
                        .reshape(batch * num_candidates, *value.shape[1:])
                    )
                q_all = self.scale_q_forward(
                    expanded, prefix.reshape(batch * num_candidates, -1), horizon
                )
                q = self._min_twin_q(q_all).reshape(batch, num_candidates)
                v = self.scale_v_forward(obs, horizon).reshape(batch, 1)
                adv = discount_normalized_advantage(
                    q, v, gamma=self.aqc_gamma, horizon=horizon
                )
                scores.append(zscore(adv, dim=-1))
        score_stack = torch.stack(scores, dim=-1)
        sample_idx, horizon_idx = select_adaptive_chunk(score_stack)
        rows = torch.arange(batch, device=candidates.device)
        best = candidates[rows, sample_idx]
        chosen_h = torch.as_tensor(
            horizons, device=candidates.device, dtype=torch.long
        )[horizon_idx]
        metrics = {
            "aqc/num_candidates": float(num_candidates),
            "aqc/chosen_horizon_mean": chosen_h.float().mean().item(),
            "aqc/chosen_full_chunk_rate": (
                (chosen_h == self.chunk_len).float().mean().item()
            ),
        }
        return best, metrics

    def sac_q_forward(
        self,
        obs,
        actions,
        shared_feature=None,
        detach_encoder=False,
        return_sequence: bool = False,
    ):
        del shared_feature
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        if not self.use_tsac_critic:
            return self.q_head(critic_state, self._flatten_batch(actions))

        action_seq = self._reshape_actions_for_tsac(actions)
        q_seq = self.q_head(critic_state, action_seq)
        if return_sequence:
            return q_seq
        # Actor / single-chunk bootstrap uses the final prefix position.
        return q_seq[:, -1, :]

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        if self.use_tsac_critic:
            raise NotImplementedError(
                "TSAC Transformer critic does not support crossq_q_forward."
            )
        del shared_feature
        critic_state = self._critic_state(obs)
        next_critic_state = (
            self._critic_state(next_obs) if next_obs is not None else None
        )
        if detach_encoder:
            critic_state = critic_state.detach()
            if next_critic_state is not None:
                next_critic_state = next_critic_state.detach()
        return self.q_head(
            critic_state,
            self._flatten_batch(actions),
            next_state_features=next_critic_state,
            next_action_features=(
                self._flatten_batch(next_actions) if next_actions is not None else None
            ),
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def sft_forward(self, data, **kwargs):
        obs = data["obs"] if "obs" in data else data
        target_actions = self._flatten_batch(
            data["action"] if "action" in data else data["actions"]
        )
        actor_state = self._actor_state(obs)
        pred_actions = self.actor_mean(self.backbone(actor_state))
        return F.mse_loss(pred_actions, target_actions, reduction="none")

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs=env_obs)
        extract_metrics: dict[str, float] = {}
        if self.action_extract == "best_of_n":
            action, extract_metrics = self._best_of_n_actions(obs)
            _, chunk_logprobs, _ = self.sac_forward(
                obs, deterministic=(mode == "eval")
            )
        elif self.action_extract == "adaptive":
            action, extract_metrics = self._adaptive_actions(obs)
            _, chunk_logprobs, _ = self.sac_forward(
                obs, deterministic=(mode == "eval")
            )
        else:
            action, chunk_logprobs, _ = self.sac_forward(
                obs, deterministic=(mode == "eval")
            )
        chunk_actions = self._format_chunk_actions(action)

        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(obs)

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": torch.zeros_like(chunk_logprobs[..., :1]),
            "forward_inputs": forward_inputs,
        }
        if extract_metrics:
            result["qc_extract_metrics"] = extract_metrics
        return chunk_actions, result
