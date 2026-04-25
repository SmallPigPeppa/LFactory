# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from itertools import chain
from typing import Any

from .processor_utils import DatasetProcessor


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        eos_token = self.tokenizer.eos_token or "<|end_of_text|>"
        text_examples = []
        for prompt, response in zip(examples["_prompt"], examples["_response"]):
            messages = (prompt or []) + (response or [])
            text = "\n".join(message["content"] for message in messages if message.get("content"))
            if text:
                text_examples.append(text + eos_token)

        if not self.data_args.packing:
            if getattr(self.tokenizer, "add_bos_token", False):
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]
            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
        else:
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.data_args.cutoff_len
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if getattr(self.tokenizer, "add_bos_token", False):
                for i in range(len(result["input_ids"])):
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id
        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
