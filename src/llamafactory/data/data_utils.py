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
    train_dict, predict_dict = {}, {}
    if dataset is not None:
        train_dict["train"] = dataset
    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, data in eval_dataset.items():
                predict_dict[f"predict_{name}"] = data
        else:
            predict_dict["predict"] = eval_dataset
    return train_dict, predict_dict


def get_dataset_module(dataset):
    module = {}
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            module["train_dataset"] = dataset["train"]
        if "predict" in dataset:
            module["predict_dataset"] = dataset["predict"]
        else:
            predict_dataset = {k[len("predict_") :]: v for k, v in dataset.items() if k.startswith("predict_")}
            if predict_dataset:
                module["predict_dataset"] = predict_dataset
            elif "validation" in dataset:
                module["predict_dataset"] = dataset["validation"]
            else:
                legacy_predict_dataset = {k[len("validation_") :]: v for k, v in dataset.items() if k.startswith("validation_")}
                if legacy_predict_dataset:
                    module["predict_dataset"] = legacy_predict_dataset
    else:
        module["train_dataset"] = dataset
    return module
