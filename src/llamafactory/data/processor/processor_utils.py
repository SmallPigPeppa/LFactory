import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetProcessor(ABC):
    template: Any
    tokenizer: Any
    processor: Any
    data_args: Any

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        ...


def search_for_fit(numbers, capacity):
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else index - 1


def greedy_knapsack(numbers, capacity):
    numbers.sort()
    knapsacks = []
    while numbers:
        current, remaining = [], capacity
        while True:
            index = search_for_fit(numbers, remaining)
            if index == -1:
                break
            remaining -= numbers[index]
            current.append(numbers.pop(index))
        knapsacks.append(current)
    return knapsacks


def infer_seqlen(source_len, target_len, cutoff_len):
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * target_len / (source_len + target_len))
    new_target_len = min(max_target_len, target_len)
    new_source_len = min(max(cutoff_len - new_target_len, 0), source_len)
    return new_source_len, new_target_len
