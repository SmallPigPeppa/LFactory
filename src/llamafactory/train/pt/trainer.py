"""PT Lightning training compatibility module."""

from ..lightning_module import LlamaFactoryLightningModule as CustomTrainer

__all__ = ["CustomTrainer"]
