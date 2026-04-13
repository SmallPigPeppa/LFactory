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

"""HyperParallel distributed trainer for LlamaFactory."""

import json
import logging
import os
import types
from contextlib import nullcontext
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import Seq2SeqTrainer

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import HSDPModule, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_ as hp_clip_grad_norm_
from hyper_parallel.integration.llamafactory import (
    export_to_hf_format,
    fsdp2_prepare_model,
    load_hsdp_model,
    load_hsdp_optimizer_and_scheduler,
    save_hsdp_checkpoint,
    wrap_optimizer_with_skip_dtensor_dispatch,
)
from hyper_parallel.integration.llamafactory.args import HyperParallelArguments

logger = logging.getLogger(__name__)


class HyperParallelTrainer(Seq2SeqTrainer):
    """Trainer that replaces Accelerate FSDP2 with HyperParallel fully_shard."""

    def __init__(
        self,
        hp_args: HyperParallelArguments,
        finetuning_args=None,
        processor=None,
        ref_model: Optional[nn.Module] = None,
        **kwargs,
    ):
        kwargs["processing_class"] = kwargs.pop("tokenizer", kwargs.get("processing_class", None))
        gen_kwargs = kwargs.pop("gen_kwargs", None)
        self._hp_args = hp_args
        self.finetuning_args = finetuning_args
        super().__init__(**kwargs)
        if not getattr(self.accelerator, "is_fsdp2", False):
            raise ValueError("HyperParallel trainer requires Accelerate FSDP2 mode to be enabled.")
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs
        self.ref_model = ref_model

        if processor is not None:
            self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model, self._hp_args)
        self._orig_accelerator_clip_grad_norm = self.accelerator.clip_grad_norm_
        self._orig_fsdp2_prepare_model = None
        self._accelerator_patches_active = False

    def _activate_accelerator_patches(self) -> None:
        """Patch Accelerate to use HyperParallel fsdp2_prepare_model and clip_grad_norm_."""
        if self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        hp_args = self._hp_args

        self._orig_fsdp2_prepare_model = acc_module.fsdp2_prepare_model

        def _hp_fsdp2_prepare_model(accelerator, model):
            return fsdp2_prepare_model(accelerator, model, hp_args)

        acc_module.fsdp2_prepare_model = _hp_fsdp2_prepare_model

        def _hp_clip_grad_norm(accelerator, parameters, max_norm, norm_type=2):
            if getattr(accelerator, "is_fsdp2", False):
                accelerator.unscale_gradients()
                parameter_list = list(parameters)
                parameter_ids = {id(param) for param in parameter_list}
                for model in accelerator._models:  # pylint: disable=protected-access
                    if not isinstance(model, HSDPModule):
                        continue
                    model_param_ids = {id(param) for param in model.parameters()}
                    if parameter_ids and parameter_ids.issubset(model_param_ids):
                        return hp_clip_grad_norm_(parameter_list, max_norm, norm_type=norm_type)
            return self._orig_accelerator_clip_grad_norm(parameters, max_norm, norm_type=norm_type)

        self.accelerator.clip_grad_norm_ = types.MethodType(_hp_clip_grad_norm, self.accelerator)
        self._accelerator_patches_active = True

    def _restore_accelerator_patches(self) -> None:
        """Restore original Accelerate methods."""
        if not self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        if self._orig_fsdp2_prepare_model is not None:
            acc_module.fsdp2_prepare_model = self._orig_fsdp2_prepare_model
        self.accelerator.clip_grad_norm_ = self._orig_accelerator_clip_grad_norm
        self._accelerator_patches_active = False

    def _wrap_model(self, model: nn.Module, training: bool = True, dataloader=None) -> nn.Module:
        """Let Accelerate own FSDP2/HSDP wrapping so optimizer remapping stays correct."""
        del dataloader
        if isinstance(model, HSDPModule):
            return model
        if training and getattr(self.accelerator, "is_fsdp2", False):
            return model
        return super()._wrap_model(model, training=training)

    def _get_train_sampler(self, *args, **kwargs):
        """Respect disable_shuffling when provided by the caller."""
        if getattr(self.finetuning_args, "disable_shuffling", False):
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, *args, **kwargs):
        """Support ASFT-style loss when a reference model is configured."""
        if getattr(self.finetuning_args, "use_asft_loss", False) and self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        return super().compute_loss(model, inputs, *args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Remove the prompt span from generated tokens during generation-based eval."""
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(self, dataset, predict_results, skip_special_tokens: bool = True) -> None:
        """Save generation results to ``generated_predictions.jsonl``."""
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info("Saving prediction results to %s", output_prediction_file)

        labels = np.where(
            predict_results.label_ids != getattr(self.data_collator, "label_pad_token_id", -100),
            predict_results.label_ids,
            self.processing_class.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != getattr(self.data_collator, "label_pad_token_id", -100),
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for index, pred in enumerate(preds):
            pad_len = np.nonzero(pred != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[index] = np.concatenate((pred[pad_len[0] :], pred[: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as file:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                file.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    def _move_model_to_device(self, model: nn.Module, device: Optional[torch.device] = None):
        """Skip redundant device moves for HSDP-wrapped models."""
        if isinstance(model, HSDPModule):
            return model
        if device is None:
            return model
        return model.to(device)

    def train(self, *args, **kwargs):
        """Activate HP patches during training and restore afterwards."""
        self._activate_accelerator_patches()
        try:
            return super().train(*args, **kwargs)
        finally:
            self._restore_accelerator_patches()

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Standard training step with HSDP gradient synchronization."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        sync_gradients = getattr(self.accelerator, "sync_gradients", True)
        if isinstance(model, HSDPModule):
            model.set_is_last_backward(sync_gradients)
            model.set_requires_gradient_sync(sync_gradients)

        compute_loss_context_manager = getattr(self, "compute_loss_context_manager", nullcontext)
        with compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not getattr(self, "model_accepts_loss_kwargs", False) and getattr(self, "compute_loss_func", None) is None:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if isinstance(model, HSDPModule) and sync_gradients:
            hsdp_sync_stream()

        return loss.detach()

    def create_optimizer(self):
        """Create optimizer and wrap step with SkipDTensorDispatch."""
        optimizer = super().create_optimizer()
        wrap_optimizer_with_skip_dtensor_dispatch(optimizer)
        return optimizer

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        """Save model/optimizer shards per-rank and scheduler."""
        save_hsdp_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            output_dir=output_dir,
            should_save_scheduler=self.args.should_save and self.lr_scheduler is not None,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Optional[nn.Module] = None) -> None:
        """Load model from HSDP sharded checkpoint."""
        target = model if model is not None else self.model
        loaded = load_hsdp_model(target, resume_from_checkpoint)
        if not loaded:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._pending_hsdp_checkpoint = resume_from_checkpoint
        return None

    def _load_optimizer_and_scheduler(self, checkpoint: Optional[str] = None) -> None:
        """Load optimizer/scheduler from per-rank checkpoint files."""
        ckpt_dir = getattr(self, "_pending_hsdp_checkpoint", None) or checkpoint
        if ckpt_dir is None:
            return
        load_hsdp_optimizer_and_scheduler(self.optimizer, self.lr_scheduler, ckpt_dir)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model weights in HuggingFace-compatible format."""
        save_dir = output_dir or self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        export_to_hf_format(self.model, getattr(self, "processing_class", None), save_dir)
