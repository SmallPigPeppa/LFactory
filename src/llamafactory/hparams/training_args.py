from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    overwrite_output_dir: bool = field(default=False)
