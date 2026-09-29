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

from rlinf.algorithms.rlt.ac_memory import ActorCorrectionBank
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.modules.rlt_mem_write import RLTLoopEncoder


def _load_stage1_mem(module: torch.nn.Module, ckpt_path: str) -> None:
    """Load the stage-1 ``mem`` module from an actor checkpoint."""
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        _normalize_wrapper_state_dict,
        resolve_full_weights,
    )
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    weights_path = resolve_full_weights(ckpt_path)
    if weights_path is None:
        raise FileNotFoundError(
            f"Stage-1 RLT checkpoint has no full_weights.pt under {ckpt_path}."
        )
    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state = _normalize_wrapper_state_dict(as_state_dict(loaded))
    mem_state = {
        key[len("mem.") :]: value
        for key, value in state.items()
        if key.startswith("mem.")
    }
    if not mem_state:
        raise RuntimeError(f"{weights_path} has no mem weights.")
    module.load_state_dict(mem_state, strict=True)


class RLTMLPPolicy(MLPPolicy):
    """MLP actor-critic policy for RLT Stage 2 heads.

    Actor input is reference chunk + RL token + proprio, including when
    ``use_mem=True``. With ``mem_ckpt``, memory is the frozen stage-1 ``mem``
    module and writes ``z`` from the returned prefix. Without a
    checkpoint, memory is a Pre-LN stack of the same depth as the RLT encoder.
    ``z + mean(I_t)`` is the first block's input.
    ``mem_scheme=1`` repeats residual cross-attention plus a feed-forward.
    ``mem_scheme=2`` alternates that cross block with a self-attention block
    on ``[tokens; z]``; the next block receives only ``z``. Training draws a
    loop length in ``[mem_len_min, mem_unroll_len]`` chunks.
    The critic reads ``z`` plus proprio. A loaded stage-1 module stays frozen
    and is not trained with reconstruction.

    ``use_actor_mem`` is a different path. The MLP outputs a residual
    ``δ``, and the executed action is ``a_ref + Δa_mem + δ``. ``Δa_mem``
    comes from a non-parametric correction bank. Critic memory does not live
    on this module.
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
        mem_ckpt: str | None = None,
        rlt_input_dim: int = 2048,
        rlt_prefix_seq_len: int = 1024,
        rlt_mlp_ratio: float = 4.0,
        loop_prefix_len: int = 16,
        loop_num_heads: int = 8,
        loop_num_layers: int = 2,
        mem_unroll_len: int = 1,
        mem_len_min: int | None = None,
        mem_scheme: int = 1,
        use_actor_mem: bool = False,
        ac_mem_capacity: int = 4096,
        ac_mem_topk: int = 8,
        ac_mem_tau: float = 0.1,
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
        use_actor_mem = bool(use_actor_mem)
        if use_mem and use_actor_mem:
            raise ValueError(
                "use_actor_mem stores action corrections outside z. use_mem "
                "loops history into z. Enable only one of them."
            )
        loop_prefix_len = int(loop_prefix_len)

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
        self.use_actor_mem = use_actor_mem
        self.loop_prefix_len = loop_prefix_len
        self.mem_unroll_len = int(mem_unroll_len)
        self.mem_len_min = (
            self.mem_unroll_len if mem_len_min is None else int(mem_len_min)
        )
        if self.mem_len_min < 1 or self.mem_len_min > self.mem_unroll_len:
            raise ValueError(
                "mem_len_min must lie in [1, mem_unroll_len], got "
                f"{self.mem_len_min} with mem_unroll_len={self.mem_unroll_len}."
            )
        if self.fixed_std <= 0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")
        self.mem = None
        if use_mem and mem_ckpt:
            from rlinf.models.embodiment.modules.rlt_token_transformer import (
                RLTTokenTransformer,
            )

            self.mem = RLTTokenTransformer(
                input_dim=int(rlt_input_dim),
                embed_dim=z_dim,
                prefix_seq_len=int(rlt_prefix_seq_len),
                num_layers=int(loop_num_layers),
                num_heads=int(loop_num_heads),
                mlp_ratio=float(rlt_mlp_ratio),
                mem_scheme=int(mem_scheme),
            )
            _load_stage1_mem(self.mem, mem_ckpt)
            self.mem.requires_grad_(False)
            self.mem.eval()
            self.loop_encoder = None
            self.register_buffer("student_clone_ready", torch.zeros(1))
        elif use_mem:
            # Name contains "encoder" so FSDP puts the compressor and its
            # decoder on the critic optimizer with TD + reconstruction.
            self.loop_encoder = RLTLoopEncoder(
                z_dim=z_dim,
                action_dim=flat_action_dim,
                prefix_len=loop_prefix_len,
                num_heads=int(loop_num_heads),
                num_layers=int(loop_num_layers),
                mem_scheme=int(mem_scheme),
            )
            # Weight-synced latch: train collect stays on VLA until the actor
            # marks |π − ref| as small enough.
            self.register_buffer("student_clone_ready", torch.zeros(1))
        else:
            self.loop_encoder = None
        self.actor_correction_bank = None
        if use_actor_mem:
            capacity = int(ac_mem_capacity)
            self.register_buffer("actor_mem_z", torch.zeros(capacity, z_dim))
            self.register_buffer(
                "actor_mem_delta", torch.zeros(capacity, flat_action_dim)
            )
            self.register_buffer("actor_mem_return", torch.zeros(capacity))
            self.register_buffer("actor_mem_advantage", torch.zeros(capacity))
            self.register_buffer(
                "actor_mem_episode",
                torch.full((capacity,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "actor_mem_time", torch.zeros(capacity, dtype=torch.long)
            )
            self.register_buffer("actor_mem_valid", torch.zeros(capacity))
            self.actor_correction_bank = ActorCorrectionBank(
                z=self.actor_mem_z,
                delta=self.actor_mem_delta,
                returns=self.actor_mem_return,
                advantage=self.actor_mem_advantage,
                episode=self.actor_mem_episode,
                time=self.actor_mem_time,
                valid=self.actor_mem_valid,
                topk=int(ac_mem_topk),
                tau=float(ac_mem_tau),
            )
        self._rollout_mem = None
        self._rollout_prev_action = None

    def set_student_clone_ready(self, ready: bool) -> None:
        flag = getattr(self, "student_clone_ready", None)
        if flag is None:
            return
        flag.fill_(1.0 if ready else 0.0)

    def is_student_clone_ready(self) -> bool:
        flag = getattr(self, "student_clone_ready", None)
        if flag is None:
            return True
        return bool(flag.detach().reshape(-1)[0].item() > 0.5)

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

    def _crop_hist_to_random_len(self, obs: dict) -> None:
        """Keep a random suffix of the stored chunks, length in ``[min, max]``.

        The stored window is ``mem_unroll_len`` chunks. Training draws one
        length per row so the loop compresses 1 chunk or up to that many.
        Eval uses the full stored window.
        """
        if "hist_valid" not in obs or not self.training:
            return
        valid = obs["hist_valid"]
        flat = valid.reshape(valid.shape[0], -1)
        steps = int(flat.shape[-1])
        high = min(self.mem_unroll_len, steps)
        low = min(self.mem_len_min, high)
        if low >= high:
            return
        if "_mem_loop_len" not in obs:
            obs["_mem_loop_len"] = torch.randint(
                low, high + 1, (flat.shape[0],), device=flat.device
            )
        lengths = obs["_mem_loop_len"].to(device=flat.device).reshape(-1).clamp(
            max=steps
        )
        index = torch.arange(steps, device=flat.device).unsqueeze(0)
        keep = index >= (steps - lengths).unsqueeze(1)
        obs["hist_valid"] = (flat & keep).reshape(valid.shape)

    def _unroll_from_obs(
        self, obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._crop_hist_to_random_len(obs)
        hist_prefix = obs["hist_prefix_embs"]
        hist_mask = obs.get("hist_prefix_mask")
        hist_valid = obs["hist_valid"].reshape(obs["hist_valid"].shape[0], -1)
        steps = hist_valid.shape[-1]
        if hist_mask is not None:
            hist_mask = hist_mask.reshape(hist_mask.shape[0], steps, -1)
        return self.loop_encoder.unroll(hist_prefix, hist_mask, hist_valid)

    def _write_mem(self, obs: dict, *, recompute: bool | None = None) -> torch.Tensor:
        if self.mem is not None:
            if "prefix_embs" not in obs:
                raise KeyError(
                    "Stage-1 memory requires prefix_embs from extract_rlt_obs."
                )
            param = next(self.mem.parameters())
            prefix = obs["prefix_embs"].to(device=param.device, dtype=param.dtype)
            z_prev = obs.get("z_prev")
            if z_prev is not None:
                z_prev = self._flatten_batch(z_prev).to(
                    device=param.device, dtype=param.dtype
                )
            with torch.no_grad():
                z = self.mem.step_memory(prefix, obs.get("prefix_mask"), z_prev)
            return z.to(dtype=torch.float32)
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
        return self.loop_encoder.write(prefix, mask, z_prev)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mem is not None:
            self.mem.eval()
        return self

    def bind_mem(self, obs: dict, *, detach: bool = False) -> dict:
        """Recompute looped ``z`` from current prefix and previous write."""
        if self.mem is not None:
            return obs
        if not self.use_mem:
            return obs
        if obs.get("_loop_z_ready") and "z_rl" in obs:
            z = obs["z_rl"]
        else:
            z = self._write_mem(obs)
            obs["_loop_z_ready"] = True
            obs["z_rl"] = z
        if detach:
            z = z.detach()
        bound = dict(obs)
        bound["z_rl"] = z
        return bound

    def reconstruction_loss(
        self, obs: dict
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Decode ``z`` onto the full image tokens. The loss uses the top-k mask."""
        if self.loop_encoder is None:
            raise RuntimeError("reconstruction_loss requires use_mem=True.")
        if "hist_prefix_embs" in obs:
            z, z_seq, valid = self._unroll_from_obs(obs)
            hist_mask = obs.get("hist_prefix_mask")
            if hist_mask is not None:
                steps = valid.shape[-1]
                hist_mask = hist_mask.reshape(hist_mask.shape[0], steps, -1)
            recon = obs.get("hist_recon_mask")
            if recon is not None:
                recon = recon.reshape(recon.shape[0], valid.shape[-1], -1)
            loss = self.loop_encoder.sequence_reconstruction_loss(
                z_seq,
                obs["hist_prefix_embs"],
                hist_mask,
                valid,
                recon,
            )
            obs["z_rl"] = z
            obs["_loop_z_ready"] = True
        else:
            if obs.get("_loop_z_ready") and "z_rl" in obs:
                z = obs["z_rl"]
            else:
                z = self._write_mem(obs)
                obs["z_rl"] = z
                obs["_loop_z_ready"] = True
            loss = self.loop_encoder.reconstruction_loss(
                z,
                obs["prefix_embs"],
                obs.get("prefix_mask"),
                obs.get("recon_mask"),
            )
        metrics = {"recon_loss": loss.detach().item()}
        loop_len = obs.get("_mem_loop_len")
        if isinstance(loop_len, torch.Tensor):
            metrics["mem_loop_len"] = float(loop_len.float().mean().item())
        return loss, metrics

    def _actor_state(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        if self.use_mem:
            obs = self.bind_mem(obs, detach=True)
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
        pre_tanh = action_mean if deterministic else probs.rsample()
        chunk_logprobs = probs.log_prob(pre_tanh)
        residual = torch.tanh(pre_tanh)
        if not self.use_actor_mem or self.actor_correction_bank is None:
            return residual, chunk_logprobs, None
        ref_chunk = self._get_ref_chunk(obs)
        episode, time = self._actor_mem_ids(obs, residual.shape[0], residual.device)
        delta_mem = self.actor_correction_bank.retrieve_action(
            self._get_z(obs), episode, time
        )
        composed = ref_chunk + delta_mem.detach() + residual
        return composed.clamp(-1.0, 1.0), chunk_logprobs, residual

    def _actor_mem_ids(
        self, obs: dict, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        episode = obs.get("mem_episode_id")
        time = obs.get("mem_time")
        if episode is None:
            episode = torch.full((batch,), -1, device=device, dtype=torch.long)
        else:
            episode = episode.reshape(batch).to(device=device, dtype=torch.long)
        if time is None:
            time = torch.zeros(batch, device=device, dtype=torch.long)
        else:
            time = time.reshape(batch).to(device=device, dtype=torch.long)
        return episode, time

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

    def _initial_rollout_z(self, batch: int, device, dtype) -> torch.Tensor:
        if self.mem is not None:
            init = self.mem.encoder.z_init.to(device=device, dtype=dtype)
            return init.expand(batch, -1).clone()
        return self.loop_encoder.initial_mem(batch, device, dtype).clone()

    def reset_rollout_mem(self, batch: int, device, dtype) -> None:
        if self.loop_encoder is None and self.mem is None:
            return
        self._rollout_mem = self._initial_rollout_z(batch, device, dtype)
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
        if not self.use_mem or (self.loop_encoder is None and self.mem is None):
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
            init = self._initial_rollout_z(batch, device, dtype)
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
