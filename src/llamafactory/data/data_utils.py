from enum import StrEnum

from datasets import DatasetDict, concatenate_datasets

from ..extras import logging

logger = logging.get_logger(__name__)

SLOTS = list[str | set[str] | dict[str, str]]


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def merge_dataset(all_datasets, data_args, seed):
    if len(all_datasets) == 1:
        return all_datasets[0]
    return concatenate_datasets(all_datasets)


def split_dataset(dataset, eval_dataset, data_args, seed):
    train_dict, eval_dict = {}, {}
    if dataset is not None:
        if data_args.val_size > 1e-6:
            val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
            split = dataset.train_test_split(test_size=val_size, seed=seed)
            train_dict["train"] = split["train"]
            eval_dict["validation"] = split["test"]
        else:
            train_dict["train"] = dataset
    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, data in eval_dataset.items():
                eval_dict[f"validation_{name}"] = data
        else:
            eval_dict["validation"] = eval_dataset
    return train_dict, eval_dict


def get_dataset_module(dataset):
    module = {}
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            module["train_dataset"] = dataset["train"]
        if "validation" in dataset:
            module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {k[len("validation_") :]: v for k, v in dataset.items() if k.startswith("validation_")}
            if eval_dataset:
                module["eval_dataset"] = eval_dataset
    else:
        module["train_dataset"] = dataset
    return module
