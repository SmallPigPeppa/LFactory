# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, OFTConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file as safe_load_file
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras import logging
from ..extras.constants import EngineName
from .model_utils.ktransformers import get_kt_peft_model, load_kt_peft_model
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from .model_utils.visual import COMPOSITE_MODELS, get_forbidden_modules, patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Freeze")
    if hasattr(model.config, "text_config"):  # composite models
        config = getattr(model.config, "text_config")
    else:
        config = model.config

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be "
                f"divisible by `num_layer_trainable` {finetuning_args.freeze_trainable_layers}."
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])  # remove weight/bias

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)

    model_type = getattr(model.config, "model_type", None)
    if not finetuning_args.freeze_multi_modal_projector and model_type in COMPOSITE_MODELS:
        trainable_layers.append(COMPOSITE_MODELS[model_type].projector_key)

    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))


def _get_adapter_state_dict_with_key_fix(adapter_path: str, model: "PreTrainedModel") -> dict[str, torch.Tensor]:
    """Load adapter state dict and fix key mismatches for composite models.

    When adapters are trained on composite models (e.g., Qwen3.5) that have a 'language_model'
    wrapper, the saved adapter keys include 'language_model' in their path. However, when
    loading these adapters onto a base model that doesn't have this wrapper (e.g., when using
    AutoModelForCausalLM directly), the keys don't match.

    This function detects such mismatches and transforms the keys accordingly.
    """
    from peft.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME

    # Determine which weights file to load
    if os.path.exists(os.path.join(adapter_path, SAFETENSORS_WEIGHTS_NAME)):
        weights_path = os.path.join(adapter_path, SAFETENSORS_WEIGHTS_NAME)
        state_dict = safe_load_file(weights_path)
    elif os.path.exists(os.path.join(adapter_path, WEIGHTS_NAME)):
        weights_path = os.path.join(adapter_path, WEIGHTS_NAME)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        return {}  # Let PEFT handle the error

    model_type = getattr(model.config, "model_type", None)

    # Check if this is a composite model with language_model wrapper
    if model_type in COMPOSITE_MODELS:
        language_model_keys = COMPOSITE_MODELS[model_type].language_model_keys
        if "language_model" in language_model_keys:
            # The model expects 'language_model' in the keys
            # Check if adapter was trained without it (rare case)
            needs_language_model_prefix = False
            for key in state_dict.keys():
                if ".model.layers." in key and ".model.language_model." not in key:
                    # Adapter was trained without language_model wrapper but model has it
                    needs_language_model_prefix = True
                    break

            if needs_language_model_prefix:
                # Transform keys to add language_model prefix
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Replace 'base_model.model.model.layers.' with 'base_model.model.model.language_model.layers.'
                    new_key = key.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")
                    new_state_dict[new_key] = value
                return new_state_dict
    else:
        # Model is not a composite model (e.g., loaded via AutoModelForCausalLM directly)
        # Check if adapter has language_model keys that need to be stripped
        needs_language_model_strip = False
        for key in state_dict.keys():
            if ".model.language_model." in key:
                needs_language_model_strip = True
                break

        if needs_language_model_strip:
            # Transform keys to remove language_model prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                # Replace 'base_model.model.model.language_model.layers.' with 'base_model.model.model.layers.'
                new_key = key.replace("base_model.model.model.language_model.", "base_model.model.model.")
                new_state_dict[new_key] = value
            logger.info_rank0("Transformed adapter keys to match base model structure (removed 'language_model' prefix).")
            return new_state_dict

    return state_dict


def _load_peft_model_with_key_fix(
    model: "PreTrainedModel",
    adapter_path: str,
    is_trainable: bool = False,
    **kwargs
) -> "PeftModel":
    """Load PEFT model with automatic key transformation for composite models.

    This function wraps PeftModel.from_pretrained and handles key mismatches that occur
    when loading adapters trained on composite models (e.g., Qwen3.5) onto base models
    with different structures.
    """
    from peft import set_peft_model_state_dict

    # First, try standard loading
    try:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable, **kwargs)
    except (RuntimeError, ValueError) as e:
        # If standard loading fails due to key mismatch, try with key fix
        error_msg = str(e)
        if "missing adapter keys" in error_msg.lower() or "unexpected adapter keys" in error_msg.lower():
            logger.warning_rank0("Adapter key mismatch detected. Attempting to fix keys...")

            # Load the fixed state dict
            fixed_state_dict = _get_adapter_state_dict_with_key_fix(adapter_path, model)

            # Create a temporary file with fixed keys
            import tempfile
            import shutil

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Copy adapter config
                from peft import PeftConfig
                config = PeftConfig.from_pretrained(adapter_path)
                config.save_pretrained(tmp_dir)

                # Save fixed state dict
                if os.path.exists(os.path.join(adapter_path, SAFETENSORS_WEIGHTS_NAME)):
                    from safetensors.torch import save_file
                    save_file(fixed_state_dict, os.path.join(tmp_dir, SAFETENSORS_WEIGHTS_NAME))
                else:
                    torch.save(fixed_state_dict, os.path.join(tmp_dir, WEIGHTS_NAME))

                # Copy other necessary files
                for fname in ["adapter_config.json", "README.md"]:
                    src = os.path.join(adapter_path, fname)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(tmp_dir, fname))

                # Load from temporary directory
                return PeftModel.from_pretrained(model, tmp_dir, is_trainable=is_trainable, **kwargs)
        else:
            raise


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        if finetuning_args.finetuning_type == "oft":
            logger.info_rank0("Fine-tuning method: OFT")
        else:
            logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_kt:
            assert len(model_args.adapter_name_or_path) == 1, "KTransformers model only accepts a single adapter"
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        if model_args.use_kt:
            if model_args.infer_backend != EngineName.KT:
                raise ValueError(
                    "We should use ktransformers as backend to infer the adapter fine-tuned by ktransformers."
                )

        for adapter in adapter_to_merge:
            model: LoraModel = _load_peft_model_with_key_fix(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume lora training
            if model_args.use_kt:
                model = load_kt_peft_model(model_args, model)
            elif model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, finetuning_args, is_trainable=is_trainable)
            else:
                model = _load_peft_model_with_key_fix(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if model_args.use_kt:
            new_list = []
            for m in target_modules:
                if m in ("down_proj", "up_proj", "gate_proj"):
                    new_list.extend([f"mlp.{m}", f"shared_experts.{m}"])
                elif m not in ("generate_linear", "orig_module", "prefill_linear"):
                    new_list.append(m)

            target_modules[:] = new_list

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BNB
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        if finetuning_args.finetuning_type == "lora":
            peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "use_rslora": finetuning_args.use_rslora,
                "use_dora": finetuning_args.use_dora,
                "modules_to_save": finetuning_args.additional_target,
            }
        elif finetuning_args.finetuning_type == "oft":
            peft_kwargs = {
                "r": finetuning_args.oft_rank,
                "oft_block_size": finetuning_args.oft_block_size,
                "target_modules": target_modules,
                "module_dropout": finetuning_args.module_dropout,
                "modules_to_save": finetuning_args.additional_target,
            }

        if model_args.use_kt:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("KTransformers is currently not supported for OFT.")
            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            else:
                raise ValueError("KTransformers is currently only supported for LoRA.")

            model = get_kt_peft_model(model, peft_config)
            print(f"KT_model:{model}")
        elif model_args.use_unsloth:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("Unsloth is currently not supported for OFT.")

            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            if finetuning_args.pissa_init:
                if finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            elif finetuning_args.finetuning_type == "oft":
                peft_config = OFTConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            model = get_peft_model(model, peft_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantized models can only be used for the LoRA or OFT tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 (zero3 already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and is_deepspeed_zero3_enabled():
        logger.info_rank0("DeepSpeed ZeRO3 detected, remaining trainable params in float32.")
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type in ["lora", "oft"]:
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")

    return model
