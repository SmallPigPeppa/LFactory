import json
import os

import numpy as np
import torch
from transformers import Seq2SeqTrainer

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, finetuning_args, processor=None, gen_kwargs=None, **kwargs):
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if processor is not None:
            self.model_accepts_loss_kwargs = False
            self.add_callback(SaveProcessorCallback(processor))
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

    def _get_train_sampler(self, *args, **kwargs):
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        labels = inputs.pop("labels", None) if self.args.predict_with_generate else inputs.get("labels")
        loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()
        return loss, generated_tokens, labels

    def save_predictions(self, dataset, predict_results, skip_special_tokens=True):
        if not self.is_world_process_zero():
            return
        path = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {path}")
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id)
        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.processing_class.pad_token_id)
        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)
        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)
        with open(path, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
