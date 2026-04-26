from ...data import LlamaFactoryDataModule, SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...model import load_model, load_tokenizer
from ..lightning_module import LlamaFactoryLightningModule
from ..lightning_utils import (
    build_lightning_trainer,
    create_modelcard_and_push,
    log_metrics,
    predict_with_outputs,
    save_metrics,
    train_with_metrics,
    validate_with_metrics,
)

logger = get_logger(__name__)


def run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks=None):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + list(getattr(tokenizer, "additional_special_tokens_ids", []))
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    data_module = LlamaFactoryDataModule(dataset_module, data_collator, training_args, finetuning_args)
    lightning_module = LlamaFactoryLightningModule(
        model=model,
        training_args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
        stage="sft",
        gen_kwargs=gen_kwargs,
        compute_accuracy=finetuning_args.compute_accuracy,
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
        if finetuning_args.include_effective_tokens_per_second:
            metrics["effective_tokens_per_sec"] = calculate_tps(dataset_module["train_dataset"], metrics, stage="sft")
        log_metrics("train", metrics)
        save_metrics(training_args, "train", metrics)

    if training_args.do_eval:
        metrics = validate_with_metrics(trainer, lightning_module, data_module)
        log_metrics("eval", metrics)
        save_metrics(training_args, "eval", metrics)

    if training_args.do_predict:
        predictions = predict_with_outputs(trainer, lightning_module, data_module)
        predict_dataset = dataset_module.get("eval_dataset")
        if isinstance(predict_dataset, dict):
            predict_dataset = next(iter(predict_dataset.values()))
        lightning_module.save_predictions(predict_dataset, predictions, generating_args.skip_special_tokens)
        predict_output = lightning_module.collect_prediction_output(predictions)
        metrics = {"predict_samples": len(predict_output.predictions)}
        log_metrics("predict", metrics)
        save_metrics(training_args, "predict", metrics)

    create_modelcard_and_push(lightning_module, model_args, data_args, training_args, finetuning_args)
