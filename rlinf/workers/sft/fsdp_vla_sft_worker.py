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
import os
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_RLINF:
            from rlinf.data.datasets.openpi_rlinf import (
                build_openpi_rlinf_sft_dataloader,
            )

            return build_openpi_rlinf_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type in (
            SupportedModel.OPENPI,
            SupportedModel.OPENPI_RLT_PROBE,
        ):
            from rlinf.data.datasets.openpi_rlinf import (
                build_official_openpi_sft_dataloader,
            )

            return build_official_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def run_eval(self):
        """Stage-1 eRLT eval reports action-chunk MSE and keeps the best router."""
        use_erlt = bool(
            self.cfg.actor.model.get("openpi", {}).get("use_erlt", False)
        ) and bool(self.cfg.actor.model.get("openpi", {}).get("erlt_use_probe", False))
        if not use_erlt:
            return super().run_eval()
        if self.eval_data_loader is None:
            return {"eval/action_mse": float("nan")}

        max_batches = int(self.cfg.runner.get("erlt_val_batches", 166))
        losses = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(self.eval_data_loader):
                if batch_index >= max_batches:
                    break
                output = self.model(
                    forward_type=ForwardType.SFT,
                    data=batch,
                )
                loss = output["loss"] if isinstance(output, dict) else output
                losses.append(float(loss.detach().item()))
        if was_training:
            self.model.train()
        mse = float(sum(losses) / max(len(losses), 1))
        metrics = {"action_mse": mse}
        metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
        best = getattr(self, "_erlt_best_action_mse", None)
        if best is None or metrics["action_mse"] < best:
            self._erlt_best_action_mse = metrics["action_mse"]
            if self._rank == 0:
                owner = self.model
                if not hasattr(owner, "erlt_router") and hasattr(owner, "module"):
                    owner = owner.module
                path = os.path.join(
                    self.cfg.runner.logger.log_path, "erlt_router_best.pt"
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(owner.erlt_router.state_dict(), path)
        return metrics

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in (
            SupportedModel.OPENPI_RLINF,
            SupportedModel.OPENPI,
            SupportedModel.OPENPI_RLT_PROBE,
        ):
            if model_type == SupportedModel.OPENPI_RLINF:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                    is_official_openpi_sft_dataloader,
                )

                num_batches = (
                    get_official_openpi_sft_num_batches(self.data_loader)
                    if is_official_openpi_sft_dataloader(self.data_loader)
                    else len(self.data_loader)
                )
            else:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                )

                num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
