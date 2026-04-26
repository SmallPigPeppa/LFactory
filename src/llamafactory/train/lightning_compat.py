try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
except ImportError:  # pragma: no cover - compatibility with older installations
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

__all__ = ["pl", "Callback", "EarlyStopping", "CSVLogger", "TensorBoardLogger", "WandbLogger"]
