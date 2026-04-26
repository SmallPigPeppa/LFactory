from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class FinetuningArguments:
    stage: Literal["pt", "sft"] = field(default="sft")
    finetuning_type: Literal["lora", "full"] = field(default="lora")
    pure_bf16: bool = field(default=False)

    lora_rank: int = field(default=8)
    lora_alpha: int | None = field(default=None)
    lora_dropout: float = field(default=0.0)
    lora_target: str = field(default="all")
    additional_target: str | None = field(default=None)
    use_rslora: bool = field(default=False)
    use_dora: bool = field(default=False)
    create_new_adapter: bool = field(default=False)

    freeze_vision_tower: bool = field(default=True)
    freeze_multi_modal_projector: bool = field(default=True)
    freeze_language_model: bool = field(default=False)

    plot_loss: bool = field(default=False)
    compute_accuracy: bool = field(default=False)
    disable_shuffling: bool = field(default=False)
    early_stopping_steps: int | None = field(default=None)
    include_effective_tokens_per_second: bool = field(default=False)

    def __post_init__(self):
        self.lora_target = self._split(self.lora_target)
        self.additional_target = self._split(self.additional_target)
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2

    @staticmethod
    def _split(value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    def to_dict(self):
        return asdict(self)
