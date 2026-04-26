"""SFT Lightning training compatibility module."""

from ..lightning_module import LlamaFactoryLightningModule as CustomSeq2SeqTrainer

__all__ = ["CustomSeq2SeqTrainer"]
