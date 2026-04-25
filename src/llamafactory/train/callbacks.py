# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Optional

from peft import PeftModel
from transformers import ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG
from ..extras.misc import get_peak_memory, is_env_enabled


from transformers import TrainerControl, TrainerState, TrainingArguments
from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"))

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return
        model = kwargs.pop("model")
        if isinstance(model, PeftModel):
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
            setattr(model.peft_config["default"], "init_lora_weights", True)
            model.save_pretrained(pissa_init_dir, safe_serialization=getattr(args, "save_safetensors", True))
            setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return
        model = kwargs.pop("model")
        if isinstance(model, PeftModel):
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
            setattr(model.peft_config["default"], "init_lora_weights", True)
            model.save_pretrained(pissa_backup_dir, safe_serialization=getattr(args, "save_safetensors", True))
            setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
            model.save_pretrained(
                pissa_convert_dir,
                safe_serialization=getattr(args, "save_safetensors", True),
                path_initial_model_for_weight_conversion=pissa_init_dir,
            )
            model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
            model.set_adapter("default")
            setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    def __init__(self) -> None:
        self.start_time = 0.0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.do_train = False

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed = cur_time - self.start_time
        avg_time = elapsed / cur_steps if cur_steps else 0
        remaining = (self.max_steps - cur_steps) * avg_time
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed)))
        self.remaining_time = str(timedelta(seconds=int(remaining)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        log_path = os.path.join(args.output_dir, TRAINER_LOG)
        if args.should_save and os.path.exists(log_path) and getattr(args, "overwrite_output_dir", False):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(log_path)

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return
        self._timing(cur_steps=state.global_step)
        last_log = state.log_history[-1] if state.log_history else {}
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=last_log.get("loss"),
            eval_loss=last_log.get("eval_loss"),
            predict_loss=last_log.get("predict_loss"),
            accuracy=last_log.get("eval_accuracy"),
            lr=last_log.get("learning_rate"),
            epoch=last_log.get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen
        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)
        logs = {k: v for k, v in logs.items() if v is not None}
        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.do_train or not args.should_save:
            return
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)
            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                self.thread_pool.submit(
                    self._write_log,
                    args.output_dir,
                    dict(
                        current_steps=self.cur_steps,
                        total_steps=self.max_steps,
                        percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps else 100,
                        elapsed_time=self.elapsed_time,
                        remaining_time=self.remaining_time,
                    ),
                )


class ReporterCallback(TrainerCallback):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return
        payload = {
            "model_args": self.model_args.to_dict(),
            "data_args": self.data_args.to_dict(),
            "finetuning_args": self.finetuning_args.to_dict(),
            "generating_args": self.generating_args.to_dict(),
        }
        if args.report_to and "wandb" in args.report_to:
            import wandb
            wandb.config.update(payload)
        if args.report_to and "trackio" in args.report_to:
            import trackio
            trackio.config.update(payload)
