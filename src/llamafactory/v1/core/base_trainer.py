# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The definition of trainer.

Init Phase:

1. Init batch generator.
2. Init optimizer (deepspeed).
3. Shard model.
4. Init optimizer (fsdp).
5. Init lr scheduler.

Train Phase:
1. Train Loop

"""

import os
from abc import abstractmethod

import torch
import torch.nn.functional as F

from ..accelerator.helper import ReduceOp
from ..accelerator.interface import Dim, DistributedInterface
from ..config import TrainingArguments
from ..utils import logging
from ..utils.helper import compute_valid_tokens
from ..utils.types import BatchInput, HFModel, ModelOutput, Tensor, TorchDataset
from .utils.batching import BatchGenerator
from .utils.checkpoint import (
    find_latest_checkpoint,
    load_metadata,
    load_rng_state,
    mark_checkpoint_complete,
    rotate_checkpoints,
    save_metadata,
    save_rng_state,
)
from .utils.rendering import Renderer


logger = logging.get_logger(__name__)


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer: Renderer,
        train_dataset: TorchDataset,
    ) -> None:
        self.args = args
        self.model = model
        self.renderer = renderer
        self.train_dataset = train_dataset

        # info
        self.global_step = 0

        # cached variables
        self.device = DistributedInterface().current_device
        self.dp_size = DistributedInterface().get_world_size(Dim.DP)
        self.model_input_names = self.renderer.processor.model_input_names

        self._create_batch_generator()
        # Calculate num_training_steps: max_steps takes priority if set
        if self.args.max_steps is not None and self.args.max_steps > 0:
            self.num_training_steps = self.args.max_steps
        else:
            self.num_training_steps = self.args.num_train_epochs * len(self.train_batch_generator)

        if self.args.enable_activation_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})

        self._deepspeed_engine = None
        dist_name = self.args.dist_config.name if self.args.dist_config is not None else None

        if dist_name == "deepspeed":
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin

            self._deepspeed_engine = DistributedPlugin("deepspeed")(
                self.model,
                self.args.dist_config,
                num_micro_batch=self.train_batch_generator.num_micro_batch,
                micro_batch_size=self.args.micro_batch_size,
            )
            self._init_optimizer()
            self._init_lr_scheduler()
            self.model, self.optimizer, self.lr_scheduler = self._deepspeed_engine.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        else:
            # fsdp2 / DDP / no dist
            self._shard_model()
            self._init_optimizer()
            self._init_lr_scheduler()

        self._resume_epoch = 0
        if self.args.resume_from_checkpoint:
            self._resume_from_checkpoint(self.args.resume_from_checkpoint)

    @property
    def _dist_name(self) -> str | None:
        return self.args.dist_config.name if self.args.dist_config is not None else None

    def _create_batch_generator(self) -> None:
        self.train_batch_generator = BatchGenerator(
            dataset=self.train_dataset,
            renderer=self.renderer,
            micro_batch_size=self.args.micro_batch_size,
            global_batch_size=self.args.global_batch_size,
            cutoff_len=self.args.cutoff_len,
            batching_workers=self.args.batching_workers,
            batching_strategy=self.args.batching_strategy,
            seed=self.args.seed,
        )

    def _shard_model(self) -> None:
        if self.args.dist_config is None:
            if DistributedInterface().get_world_size(Dim.DP) > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP

                logger.warning_rank0(
                    "dist_config is None but distributed training is enabled; falling back to DistributedDataParallel."
                )
                device_ids = None if self.device.type == "cpu" else [self.device.index]
                self.model = DDP(self.model, device_ids=device_ids)
        else:
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin

            self.model = DistributedPlugin(self.args.dist_config.name)(
                self.model,
                self.args.dist_config,
            )

    def _init_optimizer(self) -> None:
        """Init optimizer."""
        if self.args.optim_config is None:
            _trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(_trainable_params, lr=self.args.learning_rate)
        else:
            from ..plugins.trainer_plugins.optimizer import OptimizerPlugin

            self.optimizer = OptimizerPlugin(self.args.optim_config.name)(self.model, self.args.optim_config)

    def _init_lr_scheduler(self) -> None:
        """Init lr scheduler."""
        if self.args.lr_scheduler_config is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1.0)
        else:
            from ..plugins.trainer_plugins.lr_scheduler import LRSchedulerPlugin

            self.lr_scheduler = LRSchedulerPlugin(self.args.lr_scheduler_config.name)(
                self.optimizer, self.num_training_steps, self.args.lr_scheduler_config
            )

    # ==================== Checkpoint: resolve ====================

    def _resolve_checkpoint_path(self, ckpt_path: str) -> str | None:
        """Resolve 'auto' to the latest valid checkpoint, or return the path as-is."""
        if ckpt_path == "auto":
            resolved = find_latest_checkpoint(self.args.output_dir)
            if resolved is None:
                logger.warning_rank0(
                    "resume_from_checkpoint='auto' but no valid checkpoint found in "
                    f"'{self.args.output_dir}'. Training from scratch."
                )
            else:
                logger.info_rank0(f"Auto-detected latest checkpoint: {resolved}")
            return resolved
        return ckpt_path

    # ==================== Checkpoint: save ======================

    def _save_checkpoint(self, epoch: int) -> None:
        """Save a full training checkpoint at the current global step."""
        ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        rank = DistributedInterface().get_rank()

        if rank == 0:
            save_metadata(
                ckpt_dir,
                global_step=self.global_step,
                epoch=epoch,
                num_training_steps=self.num_training_steps,
            )

        if self._dist_name == "fsdp2":
            self._save_fsdp2_states(ckpt_dir)
        elif self._dist_name == "deepspeed":
            self._deepspeed_engine.accelerator.save_state(ckpt_dir)
        else:
            self._save_standard_states(ckpt_dir)

        if self._dist_name != "deepspeed" and rank == 0:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

        torch.save(
            self.train_batch_generator.state_dict(),
            os.path.join(ckpt_dir, f"dataloader_{rank}.pt"),
        )

        if self._dist_name != "deepspeed":
            save_rng_state(ckpt_dir, rank)

        DistributedInterface().sync()

        if rank == 0:
            mark_checkpoint_complete(ckpt_dir)
            if self.args.save_total_limit is not None:
                rotate_checkpoints(self.args.output_dir, self.args.save_total_limit)

        logger.info_rank0(f"Checkpoint saved to {ckpt_dir}")

    def _save_fsdp2_states(self, ckpt_dir: str) -> None:
        """Save model and optimizer via Distributed Checkpoint (FSDP2)."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)

        model_state = get_model_state_dict(self.model, options=options)
        dcp.save(state_dict=model_state, checkpoint_id=os.path.join(ckpt_dir, "model"))

        optim_state = get_optimizer_state_dict(self.model, self.optimizer, options=options)
        dcp.save(state_dict=optim_state, checkpoint_id=os.path.join(ckpt_dir, "optimizer"))

    def _save_standard_states(self, ckpt_dir: str) -> None:
        """Save model and optimizer for DDP / single-GPU via save_pretrained."""
        rank = DistributedInterface().get_rank()
        if rank == 0:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_dir = os.path.join(ckpt_dir, "model")
            model_to_save.save_pretrained(model_dir, max_shard_size="4GB")
            self.renderer.processor.save_pretrained(model_dir)

            os.makedirs(os.path.join(ckpt_dir, "optimizer"), exist_ok=True)
            torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer", "state_dict.pt"))

    # ==================== Checkpoint: load ======================

    def _resume_from_checkpoint(self, ckpt_path: str) -> None:
        """Restore full training state from a checkpoint directory."""
        ckpt_dir = self._resolve_checkpoint_path(ckpt_path)
        if ckpt_dir is None:
            return

        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"Checkpoint directory does not exist: {ckpt_dir}")

        rank = DistributedInterface().get_rank()

        metadata = load_metadata(ckpt_dir)
        self.global_step = metadata["global_step"]
        self._resume_epoch = metadata["epoch"]

        if self._dist_name == "fsdp2":
            self._load_fsdp2_states(ckpt_dir)
        elif self._dist_name == "deepspeed":
            self._deepspeed_engine.accelerator.load_state(ckpt_dir)
        else:
            self._load_standard_states(ckpt_dir)

        if self._dist_name != "deepspeed":
            sched_path = os.path.join(ckpt_dir, "scheduler.pt")
            if os.path.exists(sched_path):
                self.lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

        dl_path = os.path.join(ckpt_dir, f"dataloader_{rank}.pt")
        if os.path.exists(dl_path):
            self.train_batch_generator.load_state_dict(torch.load(dl_path, map_location="cpu"))

        if self._dist_name != "deepspeed":
            load_rng_state(ckpt_dir, rank)

        logger.info_rank0(f"Resumed from checkpoint: step={self.global_step}, epoch={self._resume_epoch}")

    def _load_fsdp2_states(self, ckpt_dir: str) -> None:
        """Load model and optimizer from Distributed Checkpoint (FSDP2)."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)

        ckpt_model_dir = os.path.join(ckpt_dir, "model")
        model_state = get_model_state_dict(self.model, options=options)
        dcp.load(state_dict=model_state, checkpoint_id=ckpt_model_dir)
        set_model_state_dict(self.model, model_state, options=options)

        ckpt_optim_dir = os.path.join(ckpt_dir, "optimizer")
        optim_state = get_optimizer_state_dict(self.model, self.optimizer, options=options)
        dcp.load(state_dict=optim_state, checkpoint_id=ckpt_optim_dir)
        set_optimizer_state_dict(self.model, self.optimizer, optim_state, options=options)

    def _load_standard_states(self, ckpt_dir: str) -> None:
        """Load model and optimizer for DDP / single-GPU."""
        import glob as glob_module

        from safetensors.torch import load_file

        model_dir = os.path.join(ckpt_dir, "model")
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model

        is_adapter_ckpt = os.path.exists(os.path.join(model_dir, "adapter_config.json"))

        if is_adapter_ckpt:
            from peft import set_peft_model_state_dict

            adapter_file = os.path.join(model_dir, "adapter_model.safetensors")
            if not os.path.exists(adapter_file):
                adapter_file = os.path.join(model_dir, "adapter_model.bin")
                adapter_state = torch.load(adapter_file, map_location="cpu")
            else:
                adapter_state = load_file(adapter_file, device="cpu")
            set_peft_model_state_dict(model_to_load, adapter_state)
        else:
            state_dict = {}
            for f in sorted(glob_module.glob(os.path.join(model_dir, "*.safetensors"))):
                state_dict.update(load_file(f, device="cpu"))
            if not state_dict:
                for f in sorted(glob_module.glob(os.path.join(model_dir, "*.bin"))):
                    state_dict.update(torch.load(f, map_location="cpu"))
            if state_dict:
                model_to_load.load_state_dict(state_dict)
            else:
                logger.warning_rank0(f"No model weights found in {model_dir}, skipping model state restore.")

        optim_path = os.path.join(ckpt_dir, "optimizer", "state_dict.pt")
        if os.path.exists(optim_path):
            self.optimizer.load_state_dict(torch.load(optim_path, map_location=self.device))

    # ==================== Core training =========================

    def compute_log_probs(self, model: HFModel, batch: BatchInput) -> Tensor:
        """Compute log probs.

        log_probs: Tensor of shape (batch_size, seq_len - 1)
        """
        batch_size, _ = batch["labels"].shape
        model_inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k in self.model_input_names
        }
        labels = batch["labels"].to(self.device, non_blocking=True)
        outputs: ModelOutput = model(**model_inputs)
        logits = outputs.logits.float()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(shift_labels.size(0), -1)
        return -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)

    @abstractmethod
    def compute_loss(self, batch: BatchInput) -> Tensor:
        """Compute the scalar loss."""
        ...

    def fit(self) -> None:
        """Train the model."""
        self.model.train()
        for epoch in range(self._resume_epoch, self.args.num_train_epochs):
            self.train_batch_generator.set_epoch(epoch)
            for micro_batches in self.train_batch_generator:
                self.global_step += 1
                step_loss = 0
                step_valid_tokens = compute_valid_tokens(micro_batches)
                step_valid_tokens = DistributedInterface().all_reduce(step_valid_tokens, op=ReduceOp.SUM)
                num_micro = len(micro_batches)
                for i, micro_batch in enumerate(micro_batches):
                    loss = self.compute_loss(micro_batch)
                    mini_step_valid_tokens = compute_valid_tokens([micro_batch])
                    # fsdp uses mean reduction so we need to scale the loss by dp_size
                    loss = loss * mini_step_valid_tokens * self.dp_size / (step_valid_tokens + 1e-6)

                    if self._deepspeed_engine is not None:
                        # deepspeed: set sync_gradients so engine.step() only fires on last micro-batch
                        self._deepspeed_engine.accelerator.sync_gradients = i == num_micro - 1
                        self._deepspeed_engine.backward(loss)
                    else:
                        loss.backward()
                    step_loss += loss.item()

                if self._deepspeed_engine is not None:
                    # deepspeed: engine.step() already ran inside backward at the sync boundary
                    grad_norm = self._deepspeed_engine.get_grad_norm()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm).item()

                    # isfinite(): argument 'input' (position 1) must be Tensor, not float
                    if not torch.isfinite(torch.tensor(grad_norm)):  # type: ignore # pyright: ignore [reportUnknownReturnType]
                        logger.warning_rank0(f"Gradient norm is not finite: {grad_norm}")
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                step_loss, grad_norm = DistributedInterface().all_reduce([step_loss, grad_norm])
                DistributedInterface().sync()
                if DistributedInterface().get_rank() == 0:
                    print(f"Epoch {epoch}, Step {self.global_step}, Loss: {step_loss:.4f}, Grad Norm: {grad_norm:.4f}")

                if self.args.save_steps and self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint(epoch)

                # Check if max_steps is reached
                if self.global_step >= self.num_training_steps:
                    logger.info_rank0(f"Reached max_steps ({self.num_training_steps}), stopping training.")
                    return

            if self.args.save_on_epoch_end:
                already_saved = self.args.save_steps and self.global_step % self.args.save_steps == 0
                if not already_saved:
                    self._save_checkpoint(epoch)

    def save_model(self) -> None:
        """Save the model."""
        if self.args.dist_config is not None and self.args.dist_config.name in ("deepspeed", "fsdp2"):
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin

            DistributedPlugin(self.args.dist_config.name).save_model(
                self.model, self.args.output_dir, self.renderer.processor
            )
        else:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_to_save.save_pretrained(self.args.output_dir, max_shard_size="4GB")
            self.renderer.processor.save_pretrained(self.args.output_dir, max_shard_size="4GB")
            logger.info_rank0(f"Model saved to {self.args.output_dir}")
