# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass, field

from transformers import Seq2SeqTrainingArguments


@dataclass
class Fp8Arguments:
    fp8: bool = field(default=False, metadata={"help": "Enable FP8 mixed precision via Accelerate."})
    fp8_backend: str = field(default="auto", metadata={"help": "FP8 backend: auto/torchao/te/msamp."})
    fp8_enable_fsdp_float8_all_gather: bool = field(
        default=False, metadata={"help": "Enable FP8 FSDP2 all-gather optimization."}
    )


@dataclass
class TrainingArguments(Fp8Arguments, Seq2SeqTrainingArguments):
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite existing output dir."})
