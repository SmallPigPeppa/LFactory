import math
from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader, SequentialSampler

from ..train.lightning_compat import pl


class LlamaFactoryDataModule(pl.LightningDataModule):
    """LightningDataModule wrapper around LLaMA-Factory datasets and collators.

    The existing ``get_dataset`` function still performs all loading, splitting and
    tokenization. This class only converts the resulting Hugging Face Dataset objects
    into PyTorch DataLoaders that Lightning can consume.
    """

    def __init__(self, dataset_module, data_collator, training_args, finetuning_args):
        super().__init__()
        self.train_dataset = dataset_module.get("train_dataset")
        self.eval_dataset = dataset_module.get("eval_dataset")
        self.predict_dataset = dataset_module.get("predict_dataset") or self.eval_dataset
        self.data_collator = data_collator
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.eval_dataset_names = self._dataset_names(self.eval_dataset)
        self.predict_dataset_names = self._dataset_names(self.predict_dataset)

    @staticmethod
    def _dataset_names(dataset):
        if isinstance(dataset, Mapping):
            return list(dataset.keys())
        return ["validation"] if dataset is not None else []

    @staticmethod
    def _datasets_to_loaders(dataset, factory):
        if dataset is None:
            return None
        if isinstance(dataset, Mapping):
            return [factory(value, shuffle=False, is_train=False) for value in dataset.values()]
        return factory(dataset, shuffle=False, is_train=False)

    def _dataloader_kwargs(self, dataset, shuffle=False, is_train=False):
        args = self.training_args
        batch_size = args.per_device_train_batch_size if is_train else args.per_device_eval_batch_size
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            drop_last=args.dataloader_drop_last if is_train else False,
        )
        if args.dataloader_num_workers and args.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = getattr(args, "dataloader_persistent_workers", False)
            prefetch_factor = getattr(args, "dataloader_prefetch_factor", None)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor

        if is_train and self.finetuning_args.disable_shuffling:
            kwargs["sampler"] = SequentialSampler(dataset)
        else:
            kwargs["shuffle"] = shuffle
        return kwargs

    def _make_loader(self, dataset, shuffle=False, is_train=False):
        return DataLoader(**self._dataloader_kwargs(dataset, shuffle=shuffle, is_train=is_train))

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return self._make_loader(
            self.train_dataset,
            shuffle=not self.finetuning_args.disable_shuffling,
            is_train=True,
        )

    def val_dataloader(self):
        return self._datasets_to_loaders(self.eval_dataset, self._make_loader)

    def predict_dataloader(self):
        return self._datasets_to_loaders(self.predict_dataset, self._make_loader)

    def num_update_steps_per_epoch(self):
        if self.train_dataset is None:
            return 0
        args = self.training_args
        world_size = max(1, int(getattr(args, "world_size", 1) or 1))
        per_device_batch = max(1, int(args.per_device_train_batch_size))
        global_batch = per_device_batch * world_size
        if args.dataloader_drop_last:
            batches = len(self.train_dataset) // global_batch
        else:
            batches = math.ceil(len(self.train_dataset) / global_batch)
        return math.ceil(max(1, batches) / max(1, int(args.gradient_accumulation_steps)))

    def max_train_steps(self):
        args = self.training_args
        if getattr(args, "max_steps", -1) and args.max_steps > 0:
            return int(args.max_steps)
        steps_per_epoch = self.num_update_steps_per_epoch()
        if steps_per_epoch <= 0:
            return -1
        return math.ceil(float(args.num_train_epochs) * steps_per_epoch)
