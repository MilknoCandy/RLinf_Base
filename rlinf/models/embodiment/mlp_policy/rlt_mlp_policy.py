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
from rlinf.models.embodiment.modules.rlt_mem_write import RLTLoopEncoder


class RLTMLPPolicy(MLPPolicy):
    """MLP actor-critic policy for RLT Stage 2 heads.

    Default actor input is reference chunk + RL token + proprio. With
    ``use_mem=True``, ``z`` is the encoder loop
    ``Encoder([I_t; a_{t-1}; r_{t-1}; z_{t-1}])``. Rollout carries ``z``
    across chunks. Training unrolls a same-episode window so ``π`` / ``Q``
    are functions of recent ``(I, a, r)``, not one isolated critical frame.
    ``ref_chunk`` is never an encoder input; it is only a BC / pred target.
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
        use_mem: bool = False,
        loop_prefix_len: int = 64,
        loop_num_heads: int = 8,
        loop_num_layers: int = 2,
        mem_unroll_len: int = 4,
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
        use_mem = bool(use_mem)
        loop_prefix_len = int(loop_prefix_len)

        if use_mem:
            actor_obs_dim = z_dim + proprio_dim
            critic_obs_dim = z_dim + proprio_dim
        else:
            actor_obs_dim = z_dim + proprio_dim + flat_action_dim
            critic_obs_dim = z_dim + proprio_dim

        super().__init__(
            obs_dim=actor_obs_dim,
            action_dim=flat_action_dim,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
            critic_obs_dim=critic_obs_dim,
        )
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.step_action_dim = step_action_dim
        self.chunk_len = chunk_len
        self.ref_chunk_len = ref_chunk_len
        self.flat_action_dim = flat_action_dim
        self.fixed_std = float(fixed_std)
        self.use_mem = use_mem
        self.loop_prefix_len = loop_prefix_len
        self.mem_unroll_len = int(mem_unroll_len)
        if self.fixed_std <= 0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")
        if use_mem:
            # Name contains "encoder" so FSDP puts the loop on the critic
            # optimizer with TD + pred_ref, not the actor -Q step.
            self.loop_encoder = RLTLoopEncoder(
                z_dim=z_dim,
                action_dim=flat_action_dim,
                prefix_len=loop_prefix_len,
                num_heads=int(loop_num_heads),
                num_layers=int(loop_num_layers),
            )
        else:
            self.loop_encoder = None
        self._rollout_mem = None
        self._rollout_prev_action = None

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

    def _unroll_from_obs(
        self, obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_prefix = obs["hist_prefix_embs"]
        hist_mask = obs.get("hist_prefix_mask")
        hist_valid = obs["hist_valid"].reshape(obs["hist_valid"].shape[0], -1)
        steps = hist_valid.shape[-1]
        hist_action = obs["hist_action"].reshape(obs["hist_action"].shape[0], steps, -1)
        hist_reward = obs["hist_reward"].reshape(obs["hist_reward"].shape[0], steps, -1)
        if hist_mask is not None:
            hist_mask = hist_mask.reshape(hist_mask.shape[0], steps, -1)
        return self.loop_encoder.unroll(
            hist_prefix, hist_mask, hist_action, hist_reward, hist_valid
        )

    def _write_mem(self, obs: dict, *, recompute: bool | None = None) -> torch.Tensor:
        if self.loop_encoder is None:
            raise RuntimeError("write_mem requires use_mem=True.")
        if recompute is False:
            return self._get_z(obs)
        if "hist_prefix_embs" in obs:
            z, _, _ = self._unroll_from_obs(obs)
            return z
        if "prefix_embs" not in obs:
            raise KeyError("Encoder loop requires prefix_embs from extract_rlt_obs.")
        prefix = obs["prefix_embs"]
        mask = obs.get("prefix_mask")
        batch = prefix.shape[0]
        device, dtype = prefix.device, prefix.dtype
        z_prev = obs.get("z_prev")
        if z_prev is None:
            z_prev = self.loop_encoder.initial_mem(batch, device, dtype)
        else:
            z_prev = self._flatten_batch(z_prev).to(device=device, dtype=dtype)
        prev_action = obs.get("prev_action")
        if prev_action is None:
            prev_action = torch.zeros(
                batch, self.flat_action_dim, device=device, dtype=dtype
            )
        else:
            prev_action = self._flatten_batch(prev_action).to(device=device, dtype=dtype)
        prev_reward = obs.get("prev_reward")
        if prev_reward is None:
            prev_reward = torch.zeros(batch, 1, device=device, dtype=dtype)
        else:
            prev_reward = self._flatten_batch(prev_reward).to(device=device, dtype=dtype)
            if prev_reward.shape[-1] != 1:
                prev_reward = prev_reward.mean(dim=-1, keepdim=True)
        return self.loop_encoder.write(prefix, mask, z_prev, prev_action, prev_reward)

    def bind_mem(self, obs: dict, *, detach: bool = False) -> dict:
        """Recompute looped ``z`` from current prefix and previous write."""
        if not self.use_mem:
            return obs
        z = self._write_mem(obs)
        if detach:
            z = z.detach()
        bound = dict(obs)
        bound["z_rl"] = z
        return bound

    def predictive_loss(
        self,
        obs: dict,
        ref_chunk: torch.Tensor,
        reward_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if self.loop_encoder is None:
            raise RuntimeError("predictive_loss requires use_mem=True.")
        if "hist_prefix_embs" in obs:
            z, z_seq, valid = self._unroll_from_obs(obs)
            pred_ref_seq = self.loop_encoder.pred_ref(z_seq)
            hist_ref = obs["hist_ref"].to(device=pred_ref_seq.device)
            hist_ref = hist_ref.reshape(pred_ref_seq.shape[0], pred_ref_seq.shape[1], -1)
            valid = valid.to(device=pred_ref_seq.device, dtype=pred_ref_seq.dtype)
            sq = torch.square(pred_ref_seq - hist_ref).mean(dim=-1)
            ref_loss = sq.mul(valid).sum() / torch.clamp(valid.sum(), min=1.0)
            pred_ref = pred_ref_seq[:, -1]
        else:
            z = self._write_mem(obs)
            pred_ref, _ = self.loop_encoder.predict(z)
            ref_loss = F.mse_loss(pred_ref, self._flatten_batch(ref_chunk))
        _, pred_r = self.loop_encoder.predict(z)
        reward_target = self._flatten_batch(reward_target).to(dtype=pred_r.dtype)
        if reward_target.shape[-1] != 1:
            reward_target = reward_target.mean(dim=-1, keepdim=True)
        reward_loss = F.mse_loss(pred_r, reward_target)
        metrics = {
            "pred_ref_loss": ref_loss.detach().item(),
            "pred_r_loss": reward_loss.detach().item(),
            "pred_ref_abs_mean": pred_ref.detach().abs().mean().item(),
        }
        return ref_loss, reward_loss, metrics

    def _actor_state(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        if self.use_mem:
            obs = self.bind_mem(obs, detach=True)
            return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)
        ref_chunk = self._get_ref_chunk(obs)
        if apply_reference_dropout:
            ref_chunk = self._maybe_drop_reference(ref_chunk, reference_dropout_prob)
        parts = [ref_chunk, self._get_z(obs), self._get_proprio(obs)]
        return torch.cat(parts, dim=-1)

    def _critic_state(self, obs: dict) -> torch.Tensor:
        if self.use_mem:
            obs = self.bind_mem(obs, detach=False)
        return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

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
        return action, chunk_logprobs, None

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        del shared_feature
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        return self.q_head(critic_state, self._flatten_batch(actions))

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
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

    def _chunk_reward_scalar(
        self, rewards: torch.Tensor | None, batch: int, device, dtype
    ) -> torch.Tensor:
        if rewards is None:
            return torch.zeros(batch, 1, device=device, dtype=dtype)
        reward = rewards.to(device=device, dtype=dtype).reshape(batch, -1)
        return reward.sum(dim=-1, keepdim=True)

    def _done_mask(
        self, dones: torch.Tensor | None, batch: int, device
    ) -> torch.Tensor:
        if dones is None:
            return torch.zeros(batch, 1, device=device, dtype=torch.bool)
        return dones.to(device=device).reshape(batch, -1).any(dim=-1, keepdim=True)

    def reset_rollout_mem(self, batch: int, device, dtype) -> None:
        if self.loop_encoder is None:
            return
        self._rollout_mem = self.loop_encoder.initial_mem(batch, device, dtype).clone()
        self._rollout_prev_action = torch.zeros(
            batch, self.flat_action_dim, device=device, dtype=dtype
        )

    def commit_rollout_action(self, actions: torch.Tensor) -> None:
        if self._rollout_prev_action is None:
            return
        flat = self._flatten_batch(actions).to(
            device=self._rollout_prev_action.device,
            dtype=self._rollout_prev_action.dtype,
        )
        self._rollout_prev_action = flat

    def _prepare_rollout_mem(
        self,
        obs: dict,
        *,
        dones: torch.Tensor | None,
        rewards: torch.Tensor | None,
    ) -> dict:
        if not self.use_mem or self.loop_encoder is None:
            return obs
        if "prefix_embs" not in obs:
            raise KeyError("Encoder loop requires prefix_embs from extract_rlt_obs.")
        prefix = obs["prefix_embs"]
        batch = prefix.shape[0]
        device, dtype = prefix.device, prefix.dtype
        if (
            self._rollout_mem is None
            or self._rollout_mem.shape[0] != batch
            or self._rollout_mem.device != device
        ):
            self.reset_rollout_mem(batch, device, dtype)
        done_mask = self._done_mask(dones, batch, device)
        if done_mask.any():
            init = self.loop_encoder.initial_mem(batch, device, dtype)
            self._rollout_mem = torch.where(done_mask, init, self._rollout_mem)
            self._rollout_prev_action = torch.where(
                done_mask,
                torch.zeros_like(self._rollout_prev_action),
                self._rollout_prev_action,
            )
        prev_reward = self._chunk_reward_scalar(rewards, batch, device, dtype)
        prev_reward = torch.where(done_mask, torch.zeros_like(prev_reward), prev_reward)
        obs = dict(obs)
        obs["z_prev"] = self._rollout_mem
        obs["prev_action"] = self._rollout_prev_action
        obs["prev_reward"] = prev_reward
        z = self._write_mem(obs, recompute=True)
        obs["z_rl"] = z
        self._rollout_mem = z.detach()
        return obs

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        dones=None,
        rewards=None,
        **kwargs,
    ):
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs=env_obs)
        obs = self._prepare_rollout_mem(obs, dones=dones, rewards=rewards)
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
        return chunk_actions, result
