# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and Optimum library.
# https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/utils/quantization_config.py
# https://github.com/huggingface/optimum/blob/v1.20.0/optimum/gptq/data.py
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

from typing import Any

import torch
from transformers import BitsAndBytesConfig, EetqConfig, HqqConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ...extras import logging
from ...extras.constants import QuantizationMethod
from ...extras.misc import get_current_device


from transformers import PretrainedConfig, PreTrainedTokenizer

from ...hparams import ModelArguments


logger = logging.get_logger(__name__)



def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    init_kwargs: dict[str, Any],
) -> None:
    r"""Priority: PTQ-quantized checkpoints > on-the-fly train-time quantization."""
    if getattr(config, "quantization_config", None):  # ptq
        if model_args.quantization_bit is not None:
            logger.warning_rank0("`quantization_bit` will not affect on the PTQ-quantized models.")

        quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
        quant_method = quantization_config.get("quant_method", "")

        if quant_method not in (QuantizationMethod.MXFP4, QuantizationMethod.FP8) and (
            is_deepspeed_zero3_enabled() or is_fsdp_enabled()
        ):
            # mxfp4 will dequant the model weights
            raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")

        if quant_method == QuantizationMethod.MXFP4:
            from transformers import Mxfp4Config

            quant_config = Mxfp4Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        if quant_method == QuantizationMethod.FP8:
            from transformers import FineGrainedFP8Config

            quant_config = FineGrainedFP8Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        if quant_method == QuantizationMethod.GPTQ:
            quantization_config.pop("disable_exllama", None)  # remove deprecated args
            quantization_config["use_exllama"] = False  # disable exllama

        if quant_method == QuantizationMethod.AQLM:
            quantization_config["bits"] = 2

        quant_bits = quantization_config.get("bits", "?")
        logger.info_rank0(f"Loading {quant_bits}-bit {quant_method.upper()}-quantized model.")


    elif model_args.quantization_bit is not None:  # on-the-fly
        if model_args.quantization_method == QuantizationMethod.BNB:
            if model_args.quantization_bit == 8:
                init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif model_args.quantization_bit == 4:
                init_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_args.compute_dtype,
                    bnb_4bit_use_double_quant=model_args.double_quantization,
                    bnb_4bit_quant_type=model_args.quantization_type,
                    bnb_4bit_quant_storage=model_args.compute_dtype,  # crucial for fsdp+qlora
                )
            else:
                raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

            # Do not assign device map if:
            # 1. deepspeed zero3 or fsdp (train)
            # 2. auto quantization device map (inference)
            if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quantization_device_map == "auto":
                if model_args.quantization_bit != 4:
                    raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")
            else:
                init_kwargs["device_map"] = {"": get_current_device()}  # change auto device map for inference

            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with bitsandbytes.")
        elif model_args.quantization_method == QuantizationMethod.HQQ:
            if model_args.quantization_bit not in [8, 6, 5, 4, 3, 2, 1]:
                raise ValueError("HQQ only accepts 1/2/3/4/5/6/8-bit quantization.")

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("HQQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            init_kwargs["quantization_config"] = HqqConfig(
                nbits=model_args.quantization_bit, quant_zero=False, quant_scale=False, axis=0
            )  # use ATEN kernel (axis=0) for performance
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with HQQ.")
        elif model_args.quantization_method == QuantizationMethod.EETQ:
            if model_args.quantization_bit != 8:
                raise ValueError("EETQ only accepts 8-bit quantization.")

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("EETQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            init_kwargs["quantization_config"] = EetqConfig()
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with EETQ.")
