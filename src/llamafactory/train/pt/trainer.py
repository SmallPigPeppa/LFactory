import torch
from transformers import Trainer

from ..callbacks import SaveProcessorCallback


class CustomTrainer(Trainer):
    def __init__(self, finetuning_args, processor=None, **kwargs):
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if processor is not None:
            self.model_accepts_loss_kwargs = False
            self.add_callback(SaveProcessorCallback(processor))

    def _get_train_sampler(self, *args, **kwargs):
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)
