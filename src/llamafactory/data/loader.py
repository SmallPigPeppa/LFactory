import os

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk

from ..extras import logging
from ..extras.misc import has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, split_dataset
from .parser import get_dataset_list
from .processor import PackedSupervisedDatasetProcessor, PretrainDatasetProcessor, SupervisedDatasetProcessor

logger = logging.get_logger(__name__)
_TOKENIZED_COLUMNS = {"input_ids", "attention_mask", "labels", "position_ids", "token_type_ids", "images", "packing_params"}


def _sanitize_tokenized_dataset(dataset):
    if isinstance(dataset, DatasetDict):
        return DatasetDict({key: _sanitize_tokenized_dataset(value) for key, value in dataset.items()})
    extra_columns = sorted(set(getattr(dataset, "column_names", []) or []) - _TOKENIZED_COLUMNS)
    if extra_columns:
        logger.warning_rank0(f"Dropping unused tokenized columns: {', '.join(extra_columns)}")
        dataset = dataset.remove_columns(extra_columns)
    return dataset


def _collect_parquet_files(path):
    if os.path.isdir(path):
        files = [os.path.join(path, name) for name in sorted(os.listdir(path)) if name.endswith(".parquet")]
    elif os.path.isfile(path) and path.endswith(".parquet"):
        files = [path]
    else:
        files = []
    if not files:
        raise ValueError(f"No parquet files found under: {path}")
    return files


def _load_single_dataset(dataset_attr, model_args, data_args, training_args):
    path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    dataset = load_dataset(
        "parquet",
        data_files=_collect_parquet_files(path),
        split=dataset_attr.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        num_proc=data_args.preprocessing_num_workers,
    )
    if dataset_attr.num_samples is not None:
        target = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target]
        if target > len(indexes):
            indexes = np.concatenate((indexes, np.random.choice(len(dataset), target - len(indexes))), axis=0)
        dataset = dataset.select(indexes)
    if data_args.max_samples is not None:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))
    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(dataset_names, model_args, data_args, training_args, return_dict=False):
    if dataset_names is None:
        return None
    datasets = {}
    for name, attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        datasets[name] = _load_single_dataset(attr, model_args, data_args, training_args)
    return datasets if return_dict else merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(data_args, stage, template, tokenizer, processor):
    if stage == "pt":
        cls = PretrainDatasetProcessor
    elif stage == "sft":
        cls = PackedSupervisedDatasetProcessor if data_args.packing else SupervisedDatasetProcessor
    else:
        raise ValueError("Only pt and sft stages are kept.")
    return cls(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _preprocess_dataset(dataset, data_args, training_args, stage, template, tokenizer, processor=None, is_eval=False):
    if dataset is None:
        return None
    dataset_processor = _get_dataset_processor(data_args, stage, template, tokenizer, processor)
    column_names = list(next(iter(dataset)).keys())
    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc="Tokenizing 779k dataset",
    )
    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration as exc:
            raise RuntimeError("No valid examples found after preprocessing.") from exc
    return dataset


def get_dataset(template, model_args, data_args, training_args, stage, tokenizer, processor=None):
    if has_tokenized_data(data_args.tokenized_path):
        logger.warning_rank0("Loading tokenized dataset from disk; raw data args are ignored.")
        tokenized_data = _sanitize_tokenized_dataset(load_from_disk(data_args.tokenized_path))
        logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
        return get_dataset_module(tokenized_data)

    with training_args.main_process_first(desc="load parquet dataset", local=(not data_args.data_shared_file_system)):
        train_dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args)
        predict_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="preprocess dataset", local=(not data_args.data_shared_file_system)):
        train_dict, predict_dict = split_dataset(train_dataset, predict_dataset, data_args, seed=training_args.seed)
        if "train" in train_dict:
            train_dict["train"] = _preprocess_dataset(train_dict["train"], data_args, training_args, stage, template, tokenizer, processor)
        for key in predict_dict:
            predict_dict[key] = _preprocess_dataset(predict_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True)
        dataset_dict = DatasetDict({**train_dict, **predict_dict})
        if data_args.tokenized_path is not None and training_args.should_save:
            dataset_dict.save_to_disk(data_args.tokenized_path)
            logger.info_rank0(f"Tokenized dataset saved at {data_args.tokenized_path}.")
        return get_dataset_module(dataset_dict)
