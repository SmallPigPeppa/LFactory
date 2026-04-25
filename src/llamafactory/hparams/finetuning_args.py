# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class FreezeArguments:
    freeze_trainable_layers: int = field(default=2, metadata={"help": "Number of trainable layers for freeze tuning."})
    freeze_trainable_modules: str = field(default="all", metadata={"help": "Comma-separated modules for freeze tuning."})
    freeze_extra_modules: str | None = field(default=None, metadata={"help": "Extra trainable modules for freeze tuning."})


@dataclass
class LoraArguments:
    additional_target: str | None = field(default=None, metadata={"help": "Extra modules to save with LoRA."})
    lora_alpha: int | None = field(default=None, metadata={"help": "LoRA alpha, defaults to 2 * rank."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_target: str = field(default="all", metadata={"help": "Comma-separated LoRA target modules, or `all`."})
    loraplus_lr_ratio: float | None = field(default=None, metadata={"help": "LoRA+ lr_B/lr_A ratio."})
    loraplus_lr_embedding: float = field(default=1e-6, metadata={"help": "LoRA+ embedding learning rate."})
    use_rslora: bool = field(default=False, metadata={"help": "Use rank-stabilized LoRA."})
    use_dora: bool = field(default=False, metadata={"help": "Use DoRA."})
    pissa_init: bool = field(default=False, metadata={"help": "Initialize a PiSSA adapter."})
    pissa_iter: int = field(default=16, metadata={"help": "FSVD steps for PiSSA; -1 means plain PiSSA."})
    pissa_convert: bool = field(default=False, metadata={"help": "Convert PiSSA adapter to normal LoRA at the end."})
    create_new_adapter: bool = field(default=False, metadata={"help": "Create a new adapter when adapter path is provided."})


@dataclass
class OFTArguments:
    additional_target: str | None = field(default=None, metadata={"help": "Extra modules to save with OFT."})
    module_dropout: float = field(default=0.0, metadata={"help": "OFT module dropout."})
    oft_rank: int = field(default=0, metadata={"help": "OFT rank."})
    oft_block_size: int = field(default=32, metadata={"help": "OFT block size."})
    oft_target: str = field(default="all", metadata={"help": "Comma-separated OFT target modules, or `all`."})
    create_new_adapter: bool = field(default=False, metadata={"help": "Create a new adapter when adapter path is provided."})


@dataclass
class FinetuningArguments(OFTArguments, LoraArguments, FreezeArguments):
    """Training knobs kept for the slim pt/sft 779k trainer."""

    pure_bf16: bool = field(default=False, metadata={"help": "Train with pure bf16 precision."})
    stage: Literal["pt", "sft"] = field(default="sft", metadata={"help": "Training stage: pt or sft."})
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(default="lora", metadata={"help": "Tuning method."})
    use_llama_pro: bool = field(default=False, metadata={"help": "Train expanded blocks only in freeze/LoRA tuning."})

    freeze_vision_tower: bool = field(default=True, metadata={"help": "Freeze the vision tower for VLM training."})
    freeze_multi_modal_projector: bool = field(default=True, metadata={"help": "Freeze the multimodal projector."})
    freeze_language_model: bool = field(default=False, metadata={"help": "Freeze the language model."})

    compute_accuracy: bool = field(default=False, metadata={"help": "Compute token-level accuracy during evaluation."})
    disable_shuffling: bool = field(default=False, metadata={"help": "Use sequential sampler for training."})
    early_stopping_steps: int | None = field(default=None, metadata={"help": "Patience for early stopping."})
    plot_loss: bool = field(default=False, metadata={"help": "Save training loss curves."})
    include_effective_tokens_per_second: bool = field(default=False, metadata={"help": "Report effective tokens/sec."})

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",") if item.strip()]
            return arg

        self.freeze_trainable_modules = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules = split_arg(self.freeze_extra_modules)
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        self.lora_target = split_arg(self.lora_target)
        self.oft_target = split_arg(self.oft_target)
        self.additional_target = split_arg(self.additional_target)
        if self.stage not in ["pt", "sft"]:
            raise ValueError("Slim build only supports `stage: pt` and `stage: sft`.")
        if self.finetuning_type not in ["lora", "oft", "freeze", "full"]:
            raise ValueError("Invalid fine-tuning method.")
        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")
        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")
            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")
            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")
            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
