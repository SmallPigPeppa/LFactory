from dataclasses import asdict, dataclass, field


@dataclass
class DataArguments:
    template: str | None = field(default=None)
    dataset: str | None = field(default=None)
    eval_dataset: str | None = field(default=None)
    dataset_dir: str = field(default="data")
    media_dir: str | None = field(default=None)
    cutoff_len: int = field(default=2048)
    train_on_prompt: bool = field(default=False)
    mask_history: bool = field(default=False)
    preprocessing_batch_size: int = field(default=1000)
    preprocessing_num_workers: int | None = field(default=None)
    overwrite_cache: bool = field(default=False)
    max_samples: int | None = field(default=None)
    val_size: float = field(default=0.0)
    eval_num_beams: int | None = field(default=None)
    eval_on_each_dataset: bool = field(default=False)
    ignore_pad_token_for_loss: bool = field(default=True)
    packing: bool | None = field(default=None)
    neat_packing: bool = field(default=False)
    default_system: str | None = field(default=None)
    enable_thinking: bool | None = field(default=True)
    preserve_thinking: bool = field(default=False)
    tokenized_path: str | None = field(default=None)
    data_shared_file_system: bool = field(default=False)

    def __post_init__(self):
        self.dataset = self._split(self.dataset)
        self.eval_dataset = self._split(self.eval_dataset)
        self.media_dir = self.media_dir or self.dataset_dir
        if self.neat_packing:
            self.packing = True
        if self.packing:
            self.cutoff_len -= 1

    @staticmethod
    def _split(value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    def to_dict(self):
        return asdict(self)
