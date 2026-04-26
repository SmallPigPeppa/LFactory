import json
import os
import shutil
import time

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ..extras.constants import TRAINER_LOG
from .lightning_compat import Callback


def _rank0():
    return int(os.getenv("LOCAL_RANK", "0")) == 0


def _jsonify(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().item() if value.numel() == 1 else value.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


class SavePretrainedCallback(Callback):
    """Save HF/PEFT ``save_pretrained`` checkpoints from Lightning training."""

    def __init__(self, training_args):
        self.training_args = training_args
        self._saved_steps = set()

    def _save_dir(self, step):
        return os.path.join(self.training_args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")

    def _rotate_checkpoints(self):
        limit = getattr(self.training_args, "save_total_limit", None)
        if not limit or limit <= 0:
            return
        output_dir = self.training_args.output_dir
        if not os.path.isdir(output_dir):
            return
        checkpoints = []
        for name in os.listdir(output_dir):
            if not name.startswith(f"{PREFIX_CHECKPOINT_DIR}-"):
                continue
            suffix = name.rsplit("-", 1)[-1]
            if suffix.isdigit():
                checkpoints.append((int(suffix), os.path.join(output_dir, name)))
        checkpoints.sort()
        while len(checkpoints) > limit:
            _, path = checkpoints.pop(0)
            shutil.rmtree(path, ignore_errors=True)

    def _save_checkpoint(self, trainer, pl_module, step):
        if step <= 0 or step in self._saved_steps or not trainer.is_global_zero:
            return
        checkpoint_dir = self._save_dir(step)
        os.makedirs(checkpoint_dir, exist_ok=True)
        pl_module.save_pretrained(checkpoint_dir)
        if not getattr(self.training_args, "save_only_model", False):
            trainer.save_checkpoint(os.path.join(checkpoint_dir, "lightning.ckpt"), weights_only=False)
        self._saved_steps.add(step)
        self._rotate_checkpoints()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        save_steps = int(getattr(self.training_args, "save_steps", 0) or 0)
        if save_steps > 0 and trainer.global_step > 0 and trainer.global_step % save_steps == 0:
            self._save_checkpoint(trainer, pl_module, trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            pl_module.save_pretrained(self.training_args.output_dir)


class SaveProcessorCallback(Callback):
    """Backward-compatible processor saver for external callback lists."""

    def __init__(self, processor):
        self.processor = processor

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.is_global_zero:
            self.processor.save_pretrained(os.path.join(pl_module.training_args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.global_step}"))

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.processor.save_pretrained(pl_module.training_args.output_dir)


class LogCallback(Callback):
    def __init__(self, training_args=None):
        self.training_args = training_args
        self.started_at = time.time()
        self._last_logged_step = -1

    def setup(self, trainer, pl_module, stage=None):
        args = self.training_args or getattr(pl_module, "training_args", None)
        if args is None or not trainer.is_global_zero:
            return
        path = os.path.join(args.output_dir, TRAINER_LOG)
        if getattr(args, "overwrite_output_dir", False) and os.path.exists(path):
            os.remove(path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.training_args or getattr(pl_module, "training_args", None)
        if args is None or not trainer.is_global_zero:
            return
        logging_steps = int(getattr(args, "logging_steps", 1) or 1)
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_logged_step or step % logging_steps != 0:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        metrics = {k: _jsonify(v) for k, v in trainer.callback_metrics.items()}
        metrics.update(
            global_step=step,
            epoch=_jsonify(trainer.current_epoch),
            learning_rate=_jsonify(getattr(pl_module, "learning_rate", args.learning_rate)),
            elapsed_seconds=round(time.time() - self.started_at, 2),
        )
        with open(os.path.join(args.output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        self._last_logged_step = step


class ReporterCallback(Callback):
    def __init__(self, model_args, data_args, finetuning_args, generating_args):
        self.payload = {
            "model_args": model_args.to_dict(),
            "data_args": data_args.to_dict(),
            "finetuning_args": finetuning_args.to_dict(),
            "generating_args": generating_args.to_dict(),
        }
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    def on_train_start(self, trainer, pl_module):
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "config") and trainer.is_global_zero:
            experiment.config.update(self.payload, allow_val_change=True)
