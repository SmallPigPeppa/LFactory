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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class ListwiseDatasetProcessor(DatasetProcessor):
    r"""Dataset processor for listwise ranking data (List DPO).

    Handles multiple ranked responses per prompt: a1 > a2 > a3 > ... > aN.
    The responses should be stored in ``_response`` as a list sorted from best to worst.
    """

    def _encode_single_response(
        self,
        prompt: list[dict[str, str]],
        response_msg: dict[str, str],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        max_target_len: int,
        source_len: int,
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(
            prompt + [response_msg], images, videos, audios, self.processor
        )
        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)

        if self.template.efficient_eos:
            response_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        prompt_ids = prompt_ids[:source_len]
        response_ids = response_ids[:max_target_len]

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * source_len + response_ids
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build input groups with format ``<bos> X``, ``Y1 <eos>``, ``Y2 <eos>``, ..., ``YN <eos>``.

        Each example produces N responses sorted by preference (best first).
        """
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            prompt = examples["_prompt"][i]
            responses = examples["_response"][i]
            system = examples["_system"][i]
            tools = examples["_tools"][i]
            images = examples["_images"][i] or []
            videos = examples["_videos"][i] or []
            audios = examples["_audios"][i] or []
            num_responses = len(responses)

            # Pre-compute prompt ids to determine source_len and max target len
            first_messages = self.template.mm_plugin.process_messages(
                prompt + [responses[0]], images, videos, audios, self.processor
            )
            prompt_ids, first_response_ids = self.template.encode_oneturn(
                self.tokenizer, first_messages, system, tools
            )
            if self.template.efficient_eos:
                first_response_ids += [self.tokenizer.eos_token_id]

            prompt_ids, _ = self.template.mm_plugin.process_token_ids(
                prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
            )

            # Find the maximum response length across all responses for consistent truncation
            max_response_len = len(first_response_ids)
            for resp in responses[1:]:
                msgs = self.template.mm_plugin.process_messages(
                    prompt + [resp], images, videos, audios, self.processor
                )
                _, resp_ids = self.template.encode_oneturn(self.tokenizer, msgs, system, tools)
                if self.template.efficient_eos:
                    resp_ids += [self.tokenizer.eos_token_id]
                max_response_len = max(max_response_len, len(resp_ids))

            source_len, target_len = infer_seqlen(
                len(prompt_ids), max_response_len, self.data_args.cutoff_len
            )

            # Encode each response
            all_input_ids = []
            all_labels = []
            for resp in responses:
                input_ids, labels = self._encode_single_response(
                    prompt, resp, system, tools, images, videos, audios, target_len, source_len
                )
                all_input_ids.append(input_ids)
                all_labels.append(labels)

            model_inputs["list_input_ids"].append(all_input_ids)
            model_inputs["list_attention_mask"].append([[1] * len(ids) for ids in all_input_ids])
            model_inputs["list_labels"].append(all_labels)
            model_inputs["list_num_responses"].append(num_responses)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        num_responses = example["list_num_responses"]
        for idx in range(num_responses):
            input_ids = example["list_input_ids"][idx]
            labels = example["list_labels"][idx]
            valid_labels = list(filter(lambda x: x != IGNORE_INDEX, labels))
            print(f"response_{idx}_input_ids:\n{input_ids}")
            print(
                f"response_{idx}_inputs:\n{self.tokenizer.decode(input_ids, skip_special_tokens=False)}"
            )
            print(f"response_{idx}_label_ids:\n{labels}")
            print(
                f"response_{idx}_labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}"
            )
