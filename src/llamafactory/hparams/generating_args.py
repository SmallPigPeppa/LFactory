from dataclasses import asdict, dataclass, field
from transformers import GenerationConfig


@dataclass
class GeneratingArguments:
    do_sample: bool = field(default=True)
    temperature: float = field(default=0.95)
    top_p: float = field(default=0.7)
    top_k: int = field(default=50)
    num_beams: int = field(default=1)
    max_length: int = field(default=1024)
    max_new_tokens: int = field(default=1024)
    repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    skip_special_tokens: bool = field(default=True)

    def to_dict(self, obey_generation_config=False):
        data = asdict(self)
        if data.get("max_new_tokens", -1) > 0:
            data.pop("max_length", None)
        else:
            data.pop("max_new_tokens", None)
        if obey_generation_config:
            ref = GenerationConfig()
            data = {k: v for k, v in data.items() if hasattr(ref, k)}
        return data
