import re
from copy import deepcopy
from dataclasses import dataclass, field

from ..extras import logging
from .data_utils import Role
from .formatter import EmptyFormatter, StringFormatter
from .mm_plugin import get_mm_plugin

logger = logging.get_logger(__name__)


@dataclass
class Template:
    format_user: StringFormatter = field(default_factory=lambda: StringFormatter(["{{content}}"]))
    format_assistant: StringFormatter = field(default_factory=lambda: StringFormatter(["{{content}}", {"eos_token"}]))
    format_system: StringFormatter = field(default_factory=lambda: StringFormatter(["{{content}}"] ))
    format_prefix: EmptyFormatter = field(default_factory=EmptyFormatter)
    default_system: str = ""
    stop_words: list[str] = field(default_factory=list)
    replace_eos: bool = False
    efficient_eos: bool = False
    enable_thinking: bool | None = True
    preserve_thinking: bool = False
    thought_words: tuple[str, str] = ("<think>\n", "\n</think>\n\n")
    mm_plugin: object = field(default_factory=lambda: get_mm_plugin("base"))

    def encode_oneturn(self, tokenizer, messages, system=None, tools=None):
        encoded = self._encode(tokenizer, messages, system)
        return sum(encoded[:-1], []), encoded[-1]

    def encode_multiturn(self, tokenizer, messages, system=None, tools=None):
        encoded = self._encode(tokenizer, messages, system)
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]

    def _slot_ids(self, tokenizer, slots):
        token_ids = []
        for slot in slots:
            if isinstance(slot, str) and slot:
                token_ids += tokenizer.encode(slot, add_special_tokens=False)
            elif isinstance(slot, set):
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    token_ids.append(tokenizer.bos_token_id)
                if "eos_token" in slot and tokenizer.eos_token_id is not None:
                    token_ids.append(tokenizer.eos_token_id)
            elif isinstance(slot, dict):
                token_ids.append(tokenizer.convert_tokens_to_ids(slot["token"]))
        return token_ids

    def _encode(self, tokenizer, messages, system=None):
        system = system or self.default_system
        encoded = []
        for idx, message in enumerate(messages):
            slots = []
            if idx == 0:
                slots += self.format_prefix.apply()
                if system:
                    slots += self.format_system.apply(content=system)

            role = message["role"]
            content = message["content"]
            if role == Role.USER or role == "user" or role == Role.OBSERVATION or role == "observation":
                slots += self.format_user.apply(content=content)
            elif role == Role.ASSISTANT or role == "assistant" or role == Role.FUNCTION or role == "function":
                slots += self.format_assistant.apply(content=content)
            else:
                raise ValueError(f"Unexpected role: {role}")
            encoded.append(self._slot_ids(tokenizer, slots))
        return encoded

    def fix_special_tokens(self, tokenizer):
        stop_words = list(self.stop_words)
        if self.replace_eos and stop_words:
            eos_token = stop_words.pop(0)
            if tokenizer.eos_token != eos_token:
                tokenizer.add_special_tokens({"eos_token": eos_token})
                logger.info_rank0(f"Set eos token: {tokenizer.eos_token}")
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if stop_words:
            tokenizer.add_special_tokens({"additional_special_tokens": stop_words}, replace_additional_special_tokens=False)

    def fix_jinja_template(self, tokenizer):
        return None

    def remove_thought(self, content):
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer):
        return tokenizer.encode(self.thought_words[0] + self.thought_words[1], add_special_tokens=False)


@dataclass
class ReasoningTemplate(Template):
    def encode_multiturn(self, tokenizer, messages, system=None, tools=None):
        messages = deepcopy(messages)
        if self.enable_thinking is False:
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])
        encoded = self._encode(tokenizer, messages, system)
        for i in range(0, len(messages), 2):
            answer = messages[i + 1]["content"]
            if "<think>" not in answer and "</think>" not in answer:
                if self.enable_thinking is False:
                    encoded[i] += self.get_thought_word_ids(tokenizer)
                else:
                    encoded[i + 1] = self.get_thought_word_ids(tokenizer) + encoded[i + 1]
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]


TEMPLATES = {}


def register_template(name, format_user=None, format_assistant=None, format_system=None, format_prefix=None, default_system="", stop_words=None, replace_eos=False, mm_plugin=None, template_class=Template):
    TEMPLATES[name] = template_class(
        format_user=format_user or StringFormatter(["{{content}}"]),
        format_assistant=format_assistant or StringFormatter(["{{content}}", {"eos_token"}]),
        format_system=format_system or StringFormatter(["{{content}}"]),
        format_prefix=format_prefix or EmptyFormatter(),
        default_system=default_system,
        stop_words=stop_words or [],
        replace_eos=replace_eos,
        mm_plugin=mm_plugin or get_mm_plugin("base"),
    )


def get_template_and_fix_tokenizer(tokenizer, data_args):
    name = data_args.template or "empty"
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template `{name}`. Kept templates: {', '.join(TEMPLATES)}")
    template = deepcopy(TEMPLATES[name])
    if data_args.default_system is not None:
        template.default_system = data_args.default_system
    if isinstance(template, ReasoningTemplate):
        template.enable_thinking = data_args.enable_thinking
        template.preserve_thinking = data_args.preserve_thinking
    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template


register_template("empty", format_assistant=StringFormatter(["{{content}}"]))

register_template(
    "llava",
    format_user=StringFormatter(["USER: {{content}} ASSISTANT:"]),
    default_system="A chat between a curious user and an artificial intelligence assistant.",
    mm_plugin=get_mm_plugin("llava", image_token="<image>"),
)

register_template(
    "llava_next",
    format_user=StringFormatter(["USER: {{content}} ASSISTANT:"]),
    default_system="A chat between a curious user and an artificial intelligence assistant.",
    mm_plugin=get_mm_plugin("llava_next", image_token="<image>"),
)

_intern_user = StringFormatter(["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"])
_intern_assistant = StringFormatter(["{{content}}<|im_end|>\n"])
_intern_system = StringFormatter(["<|im_start|>system\n{{content}}<|im_end|>\n"])
_intern_prefix = EmptyFormatter([{"bos_token"}])

register_template(
    "intern_vl",
    format_user=_intern_user,
    format_assistant=_intern_assistant,
    format_system=_intern_system,
    format_prefix=_intern_prefix,
    default_system="你是书生·万象，英文名是InternVL。",
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin("intern_vl", image_token="<image>"),
)
register_template(
    "intern_s1",
    format_user=_intern_user,
    format_assistant=_intern_assistant,
    format_system=_intern_system,
    format_prefix=_intern_prefix,
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin("intern_vl", image_token="<image>"),
)

_qwen_user = StringFormatter(["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"])
_qwen_assistant = StringFormatter(["{{content}}<|im_end|>\n"])
_qwen_system = StringFormatter(["<|im_start|>system\n{{content}}<|im_end|>\n"])

register_template(
    "qwen2_vl",
    format_user=_qwen_user,
    format_assistant=_qwen_assistant,
    format_system=_qwen_system,
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("qwen2_vl", image_token="<|image_pad|>"),
)
register_template(
    "qwen3_vl",
    format_user=_qwen_user,
    format_assistant=_qwen_assistant,
    format_system=_qwen_system,
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("qwen3_vl", image_token="<|image_pad|>"),
    template_class=ReasoningTemplate,
)
register_template(
    "qwen3_vl_nothink",
    format_user=_qwen_user,
    format_assistant=_qwen_assistant,
    format_system=_qwen_system,
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("qwen3_vl", image_token="<|image_pad|>"),
)
