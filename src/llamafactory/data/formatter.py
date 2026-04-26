from dataclasses import dataclass, field

from .data_utils import SLOTS


@dataclass
class EmptyFormatter:
    slots: SLOTS = field(default_factory=list)

    def apply(self, **kwargs):
        return self.slots


@dataclass
class StringFormatter:
    slots: SLOTS = field(default_factory=list)

    def apply(self, **kwargs):
        output = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    slot = slot.replace("{{" + name + "}}", value, 1)
            output.append(slot)
        return output


FunctionFormatter = StringFormatter


@dataclass
class ToolFormatter:
    def apply(self, **kwargs):
        return [kwargs.get("content", "")]

    def extract(self, content):
        return content
