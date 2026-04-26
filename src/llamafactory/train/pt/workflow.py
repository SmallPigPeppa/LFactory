import math

from transformers import DataCollatorForLanguageModeling

from ...data import LlamaFactoryDataModule, get_dataset, get_template_and_fix_tokenizer
from ...model import load_model, load_tokenizer
from ..lightning_module import LlamaFactoryLightningModule
from ..lightning_utils import (
    build_lightning_trainer,
    create_modelcard_and_push,
    log_metrics,
    save_metrics,
    train_with_metrics,
    validate_with_metrics,
)


def run_pt(model_args, data_args, training_args, finetuning_args, callbacks=None):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_module = LlamaFactoryDataModule(dataset_module, data_collator, training_args, finetuning_args)
    lightning_module = LlamaFactoryLightningModule(
        model=model,
        training_args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
        stage="pt",
    )
    trainer = build_lightning_trainer(
        training_args,
        finetuning_args,
        data_module,
        callbacks=callbacks,
        enable_validation_during_fit=training_args.do_eval,
    )

    if training_args.do_train:
        metrics = train_with_metrics(trainer, lightning_module, data_module, training_args)
        if "eval_loss" in metrics:
            try:
                metrics["perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["perplexity"] = float("inf")
        log_metrics("train", metrics)
        save_metrics(training_args, "train", metrics)

    if training_args.do_eval:
        metrics = validate_with_metrics(trainer, lightning_module, data_module)
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except (OverflowError, KeyError):
            if "eval_loss" in metrics:
                metrics["perplexity"] = float("inf")
        log_metrics("eval", metrics)
        save_metrics(training_args, "eval", metrics)

    create_modelcard_and_push(lightning_module, model_args, data_args, training_args, finetuning_args)
