# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import os
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk

from ..extras import logging
from ..extras.misc import has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, split_dataset
from .parser import get_dataset_list
from .processor import PackedSupervisedDatasetProcessor, PretrainDatasetProcessor, SupervisedDatasetProcessor


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _collect_parquet_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = [os.path.join(path, name) for name in sorted(os.listdir(path)) if name.endswith(".parquet")]
    elif os.path.isfile(path) and path.endswith(".parquet"):
        files = [path]
    else:
        files = []
    if not files:
        raise ValueError(f"Cannot find parquet files under: {path}")
    return files


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    logger.info_rank0(f"Loading 779k parquet dataset {dataset_attr}...")
    local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    data_files = _collect_parquet_files(local_path)
    dataset = load_dataset(
        path="parquet",
        data_files=data_files,
        split=dataset_attr.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        num_proc=data_args.preprocessing_num_workers,
        streaming=data_args.streaming,
    )
    if data_args.streaming:
        dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]
        target_num -= len(indexes)
        if target_num > 0:
            indexes = np.concatenate((indexes, np.random.choice(len(dataset), target_num)), axis=0)
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None and not data_args.streaming:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft"],
    return_dict: bool = False,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    if dataset_names is None:
        return None
    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)
    return datasets if return_dict else merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
) -> "DatasetProcessor":
    if stage == "pt":
        processor_cls = PretrainDatasetProcessor
    elif stage == "sft":
        processor_cls = PackedSupervisedDatasetProcessor if data_args.packing else SupervisedDatasetProcessor
    else:
        raise ValueError("Slim build only supports `pt` and `sft` stages.")
    return processor_cls(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"] | None,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"] | None:
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(data_args, stage, template, tokenizer, processor)
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Tokenizing 779k dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )
    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            raise RuntimeError("Cannot find valid samples in the parquet dataset.")
    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading tokenized dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()
            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module
        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving tokenized data to disk.")

    with training_args.main_process_first(desc="load parquet dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if "train" in train_dict:
            train_dict["train"] = _get_preprocessed_dataset(
                train_dict["train"], data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )
        for key in eval_dict:
            eval_dict[key] = _get_preprocessed_dataset(
                eval_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )
        dataset_dict = DatasetDict({**train_dict, **eval_dict})
        if data_args.tokenized_path is not None and training_args.should_save:
            dataset_dict.save_to_disk(data_args.tokenized_path)
            logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
        return get_dataset_module(dataset_dict)
