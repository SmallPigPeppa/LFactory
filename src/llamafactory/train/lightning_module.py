import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import get_scheduler

from ..extras import logging
from ..extras.constants import IGNORE_INDEX
from .lightning_compat import pl

logger = logging.get_logger(__name__)


@dataclass
class PredictionOutput:
    predictions: list[torch.Tensor]
    label_ids: list[torch.Tensor]
    input_ids: list[torch.Tensor]


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().float().item())
    if isinstance(value, (np.generic,)):
        return float(value)
    return value


def _numpify(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return tensor


class LlamaFactoryLightningModule(pl.LightningModule):
    """LightningModule that wraps a Hugging Face causal/vision-language model.

    The wrapped model remains the source of truth for forward, generation and
    ``save_pretrained``. Lightning owns only the training/evaluation loops,
    optimization and metric logging.
    """

    def __init__(
        self,
        model,
        training_args,
        finetuning_args,
        tokenizer=None,
        processor=None,
        stage="sft",
        gen_kwargs=None,
        compute_accuracy=False,
    ):
        super().__init__()
        self.model = model
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.tokenizer = tokenizer
        self.processor = processor
        self.stage = stage
        self.gen_kwargs = gen_kwargs or {}
        self.compute_accuracy = compute_accuracy
        self._validation_accuracy = []
        # Keep Lightning checkpoint hyperparameters compact and serializable.
        self.save_hyperparameters({"stage": stage, "compute_accuracy": compute_accuracy})

    def forward(self, **batch):
        return self.model(**batch)

    @staticmethod
    def _loss_from_outputs(outputs):
        if hasattr(outputs, "loss"):
            return outputs.loss
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise RuntimeError("The wrapped model did not return a loss. Make sure labels are included in each batch.")

    @staticmethod
    def _batch_size(batch):
        input_ids = batch.get("input_ids")
        return int(input_ids.size(0)) if torch.is_tensor(input_ids) and input_ids.dim() > 0 else None

    def _shared_loss_step(self, batch, metric_name):
        outputs = self.model(**batch)
        loss = self._loss_from_outputs(outputs)
        self.log(
            metric_name,
            loss,
            on_step=(metric_name == "train_loss"),
            on_epoch=True,
            prog_bar=(metric_name != "train_loss"),
            logger=True,
            sync_dist=True,
            batch_size=self._batch_size(batch),
        )
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_loss_step(batch, "train_loss")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.get("labels")
        loss, outputs = self._shared_loss_step(batch, "eval_loss")
        if self.compute_accuracy and labels is not None and not self.training_args.predict_with_generate:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
            if isinstance(logits, (tuple, list)):
                logits = logits[0] if logits[0].dim() == 3 else logits[1]
            preds = torch.argmax(logits.detach(), dim=-1)
            pred = preds[:, :-1]
            label = labels[:, 1:]
            mask = label != IGNORE_INDEX
            if mask.any():
                accuracy = (pred[mask] == label[mask]).float().mean()
                self._validation_accuracy.append(accuracy.detach())
                self.log(
                    "eval_accuracy",
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=self._batch_size(batch),
                )
        return {"loss": loss.detach()}

    def on_validation_epoch_end(self):
        self._validation_accuracy.clear()

    def _generation_inputs(self, batch):
        ignored = {"labels"}
        return {k: v for k, v in batch.items() if k not in ignored and v is not None}

    def _generate(self, batch):
        gen_inputs = self._generation_inputs(batch)
        generated = self.model.generate(**gen_inputs, **self.gen_kwargs)
        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids) and generated is not None and generated.dim() == 2:
            prompt_len = min(input_ids.size(-1), generated.size(-1))
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
            generated[:, :prompt_len] = pad_token_id
            generated = generated.contiguous()
        return generated

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.get("labels")
        input_ids = batch.get("input_ids")
        if self.training_args.predict_with_generate:
            predictions = self._generate(batch)
        else:
            outputs = self.model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
            predictions = torch.argmax(logits, dim=-1)
        return {
            "predictions": predictions.detach().cpu() if torch.is_tensor(predictions) else predictions,
            "labels": labels.detach().cpu() if torch.is_tensor(labels) else labels,
            "input_ids": input_ids.detach().cpu() if torch.is_tensor(input_ids) else input_ids,
        }

    def configure_optimizers(self):
        args = self.training_args
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and not name.endswith(".bias") and "norm" not in name.lower():
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None) or getattr(args, "max_steps", 0)
        total_steps = int(total_steps) if total_steps and total_steps != float("inf") else 0
        if total_steps <= 0:
            return optimizer

        warmup_steps = args.get_warmup_steps(total_steps) if hasattr(args, "get_warmup_steps") else int(args.warmup_ratio * total_steps)
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    @property
    def learning_rate(self):
        try:
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                return scheduler.get_last_lr()[0]
        except Exception:
            pass
        try:
            optimizer = self.optimizers(use_pl_optimizer=False)
            if optimizer is not None:
                return optimizer.param_groups[0]["lr"]
        except Exception:
            pass
        return self.training_args.learning_rate

    def save_pretrained(self, output_dir):
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        os.makedirs(output_dir, exist_ok=True)
        safe_serialization = getattr(self.training_args, "save_safetensors", True)
        self.model.save_pretrained(output_dir, safe_serialization=safe_serialization)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

    @staticmethod
    def _flatten_prediction_batches(predictions):
        if predictions is None:
            return []
        # Lightning returns a list per dataloader when predict_dataloader returns multiple loaders.
        if predictions and isinstance(predictions[0], list):
            flattened = []
            for loader_output in predictions:
                flattened.extend(loader_output)
            return flattened
        return predictions

    @classmethod
    def collect_prediction_output(cls, predictions):
        batches = cls._flatten_prediction_batches(predictions)
        preds, labels, inputs = [], [], []
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            p, l, i = batch.get("predictions"), batch.get("labels"), batch.get("input_ids")
            if torch.is_tensor(p):
                preds.extend([row for row in p])
            if torch.is_tensor(l):
                labels.extend([row for row in l])
            if torch.is_tensor(i):
                inputs.extend([row for row in i])
        return PredictionOutput(predictions=preds, label_ids=labels, input_ids=inputs)

    def save_predictions(self, dataset, predictions, skip_special_tokens=True):
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        if self.tokenizer is None:
            raise RuntimeError("Cannot save generated predictions without a tokenizer.")
        output = self.collect_prediction_output(predictions)
        path = os.path.join(self.training_args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {path}")
        os.makedirs(self.training_args.output_dir, exist_ok=True)

        pad_token_id = self.tokenizer.pad_token_id
        label_rows = []
        for row in output.label_ids:
            arr = _numpify(row)
            if arr is None:
                continue
            label_rows.append(np.where(arr != IGNORE_INDEX, arr, pad_token_id))
        pred_rows = []
        for row in output.predictions:
            arr = _numpify(row)
            if arr is None:
                continue
            pred_rows.append(np.where(arr != IGNORE_INDEX, arr, pad_token_id))
        input_rows = [_numpify(row) for row in output.input_ids if row is not None]

        decoded_inputs = self.tokenizer.batch_decode(input_rows, skip_special_tokens=False) if input_rows else []
        decoded_preds = self.tokenizer.batch_decode(pred_rows, skip_special_tokens=skip_special_tokens) if pred_rows else []
        decoded_labels = self.tokenizer.batch_decode(label_rows, skip_special_tokens=skip_special_tokens) if label_rows else []
        with open(path, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
