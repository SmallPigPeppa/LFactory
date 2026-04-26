import json
import os
import time

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ..extras.constants import TRAINER_LOG


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            self.processor.save_pretrained(os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"))

    def on_train_end(self, args, state, control, **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class LogCallback(TrainerCallback):
    def __init__(self):
        self.started_at = time.time()

    def on_init_end(self, args, state, control, **kwargs):
        path = os.path.join(args.output_dir, TRAINER_LOG)
        if args.should_save and os.path.exists(path) and getattr(args, "overwrite_output_dir", False):
            os.remove(path)

    def on_log(self, args, state, control, **kwargs):
        if not args.should_save or not state.log_history:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        last = dict(state.log_history[-1])
        last.update(global_step=state.global_step, elapsed_seconds=round(time.time() - self.started_at, 2))
        with open(os.path.join(args.output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(last, ensure_ascii=False) + "\n")


class ReporterCallback(TrainerCallback):
    def __init__(self, model_args, data_args, finetuning_args, generating_args):
        self.payload = {
            "model_args": model_args.to_dict(),
            "data_args": data_args.to_dict(),
            "finetuning_args": finetuning_args.to_dict(),
            "generating_args": generating_args.to_dict(),
        }
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and args.report_to and "wandb" in args.report_to:
            import wandb
            wandb.config.update(self.payload)
