from transformers import Trainer


def create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args):
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory-slim-core", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = data_args.dataset
    if training_args.do_train and training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    elif training_args.do_train:
        Trainer.create_model_card(trainer, license="other", **kwargs)
