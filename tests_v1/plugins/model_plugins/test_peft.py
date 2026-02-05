# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwareß
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, call, patch

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel

from llamafactory.v1.plugins.model_plugins import peft


class MockLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10))


class MockEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10))


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "self_attn": torch.nn.ModuleDict(
                            {
                                "q_proj": MockLinear(),
                                "k_proj": MockLinear(),
                            }
                        ),
                        "mlp": torch.nn.ModuleDict(
                            {
                                "gate_proj": MockLinear(),
                            }
                        ),
                    }
                )
            ]
        )
        self.lm_head = MockLinear()
        self.embed_tokens = MockEmbedding()
        self.config = MagicMock()
        self.config.num_hidden_layers = 1
        self.config.model_type = "llama"


def test_find_all_linear_modules():
    """Verify linear module filtering only returns allowed linear layer names."""
    model = MockModel()
    modules = peft._find_all_linear_modules(model)
    expected = {"q_proj", "k_proj", "gate_proj"}
    assert set(modules) == expected


@patch("llamafactory.v1.plugins.model_plugins.peft.PeftModel")
def test_load_adapter_single(mock_peft_model):
    """Verify a single adapter is merged and unloaded in inference mode."""
    model = MagicMock()
    mock_peft_instance = MagicMock()
    mock_peft_model.from_pretrained.return_value = mock_peft_instance
    mock_peft_instance.merge_and_unload.return_value = model
    peft.load_adapter(model, "adapter_path", is_train=False)
    mock_peft_model.from_pretrained.assert_called_with(model, "adapter_path")
    mock_peft_instance.merge_and_unload.assert_called_once()


@patch("llamafactory.v1.plugins.model_plugins.peft.PeftModel")
def test_load_adapter_resume_train_single(mock_peft_model):
    """Verify training mode loads a single adapter for continued training."""
    model = MagicMock()
    mock_peft_instance = MagicMock()
    mock_peft_model.from_pretrained.return_value = mock_peft_instance
    peft.load_adapter(model, "adapter1", is_train=True)
    mock_peft_model.from_pretrained.assert_called_once()
    assert mock_peft_model.from_pretrained.call_args[1].get("is_trainable")


@patch("llamafactory.v1.plugins.model_plugins.peft.PeftModel")
def test_load_adapter_train_multiple_disallowed(mock_peft_model):
    """Verify training mode rejects multiple adapters."""
    model = MagicMock()
    with pytest.raises(ValueError, match="only a single LoRA adapter"):
        peft.load_adapter(model, ["adapter1", "adapter2"], is_train=True)


@patch("llamafactory.v1.plugins.model_plugins.peft.PeftModel")
def test_load_adapter_infer_multiple_merges(mock_peft_model):
    model = MagicMock()
    mock_peft_instance = MagicMock()
    mock_peft_model.from_pretrained.return_value = mock_peft_instance
    mock_peft_instance.merge_and_unload.return_value = model
    peft.load_adapter(model, ["adapter1", "adapter2"], is_train=False)
    assert mock_peft_model.from_pretrained.call_args_list == [call(model, "adapter1"), call(model, "adapter2")]
    assert mock_peft_instance.merge_and_unload.call_count == 2


@patch("llamafactory.v1.plugins.model_plugins.peft.get_peft_model")
@patch("llamafactory.v1.plugins.model_plugins.peft.LoraConfig")
def test_get_lora_model(mock_lora_config, mock_get_peft_model):
    """Verify LoRA config construction and get_peft_model call arguments."""
    model = MockModel()
    config = {"r": 16, "target_modules": "all", "lora_alpha": 32}
    peft.get_lora_model(model, config, is_train=True)
    mock_lora_config.assert_called_once()
    call_kwargs = mock_lora_config.call_args[1]
    assert call_kwargs["r"] == 16
    assert set(call_kwargs["target_modules"]) == {"q_proj", "k_proj", "gate_proj"}
    mock_get_peft_model.assert_called_once()


def test_get_freeze_model_layers():
    """Verify layer-based freezing marks only target layers trainable."""
    model = MockModel()
    config = {"freeze_trainable_layers": 1, "freeze_trainable_modules": "all"}
    for param in model.parameters():
        param.requires_grad = False
    peft.get_freeze_model(model, config, is_train=True)
    for name, param in model.named_parameters():
        if "layers.0" in name:
            assert param.requires_grad
        else:
            if "embed_tokens" in name or "lm_head" in name:
                assert not param.requires_grad


def test_get_freeze_model_modules():
    """Verify module-based freezing marks only target modules trainable."""
    model = MockModel()
    config = {"freeze_trainable_layers": 1, "freeze_trainable_modules": ["self_attn"]}
    for param in model.parameters():
        param.requires_grad = False
    peft.get_freeze_model(model, config, is_train=True)
    for name, param in model.named_parameters():
        if "self_attn" in name:
            assert param.requires_grad
        elif "mlp" in name:
            assert not param.requires_grad


class FakePeftModel(torch.nn.Module):
    called_paths: list[str] = []

    def __init__(self, model, adapter_path):
        super().__init__()
        self.base_model = model
        self.adapter_path = adapter_path

    def merge_and_unload(self):
        return self.base_model

    @classmethod
    def from_pretrained(cls, model, adapter_path, **kwargs):
        cls.called_paths.append(adapter_path)
        return cls(model, adapter_path)


class DummyConfig(PretrainedConfig):
    model_type = "dummy"


class ExportModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self):
        super().__init__(DummyConfig())
        self.config.torch_dtype = torch.float32
        self.saved = None
        self.pushed = None
        self.last_to = None

    def save_pretrained(self, save_directory, **kwargs):
        self.saved = (save_directory, kwargs)

    def push_to_hub(self, repo_id):
        self.pushed = repo_id

    def to(self, *args, **kwargs):
        self.last_to = args[0] if args else kwargs.get("dtype")
        return self


@patch("llamafactory.v1.plugins.model_plugins.peft.get_args")
@patch("llamafactory.v1.plugins.model_plugins.peft.ModelEngine")
@patch("llamafactory.v1.plugins.model_plugins.peft.PeftModel", new=FakePeftModel)
def test_merge_and_export_model_success(mock_model_engine, mock_get_args):
    """Verify export saves model and tokenizer and converts dtype."""
    FakePeftModel.called_paths = []
    mock_model_args = MagicMock()
    mock_model_args.peft_config = {
        "name": "lora",
        "adapter_name_or_path": ["path1"],
        "export_dir": "/tmp/export",
        "export_size": 1,
        "infer_dtype": "float16",
        "export_legacy_format": False,
    }

    mock_get_args.return_value = (mock_model_args, None, None, None)

    mock_model = ExportModel()
    mock_tokenizer = MagicMock()
    mock_tokenizer.padding_side = "right"

    mock_model_engine_instance = MagicMock()
    mock_model_engine_instance.model = mock_model
    mock_model_engine_instance.processor = mock_tokenizer
    mock_model_engine.return_value = mock_model_engine_instance
    peft.merge_and_export_model()
    assert mock_model.last_to == torch.float16
    assert mock_model.saved[0] == "/tmp/export"
    assert mock_model.saved[1]["max_shard_size"] == "1GB"
    assert mock_model.saved[1]["safe_serialization"] is True
    mock_tokenizer.save_pretrained.assert_called_with("/tmp/export")
    assert mock_tokenizer.padding_side == "left"


@patch("llamafactory.v1.plugins.model_plugins.peft.get_args")
def test_merge_and_export_model_no_export_dir(mock_get_args):
    """Verify missing export_dir raises an error."""
    mock_model_args = MagicMock()
    mock_model_args.peft_config = {"name": "lora"}
    mock_get_args.return_value = (mock_model_args, None, None, None)
    with pytest.raises(ValueError, match="Please specify export_dir"):
        peft.merge_and_export_model()


@patch("llamafactory.v1.plugins.model_plugins.peft.get_args")
@patch("llamafactory.v1.plugins.model_plugins.peft.ModelEngine")
@patch("llamafactory.v1.plugins.model_plugins.peft.PeftModel", new=FakePeftModel)
def test_merge_and_export_model_with_adapter_path(mock_model_engine, mock_get_args):
    """Verify adapter paths are passed to model engine and the model is pushed to hub."""
    mock_model_args = MagicMock()
    mock_model_args.peft_config = {
        "name": "lora",
        "adapter_name_or_path": ["path1", "path2"],
        "export_dir": "/tmp/export",
        "export_size": 1,
        "infer_dtype": "float16",
        "export_legacy_format": False,
        "export_hub_model_id": "hub_id",
    }

    mock_get_args.return_value = (mock_model_args, None, None, None)

    mock_model = ExportModel()
    mock_model.push_to_hub = MagicMock()

    mock_model_engine_instance = MagicMock()
    mock_model_engine_instance.model = mock_model
    mock_model_engine_instance.processor = MagicMock()
    mock_model_engine.return_value = mock_model_engine_instance
    peft.merge_and_export_model()
    assert mock_model_engine.call_args[0][0].peft_config["adapter_name_or_path"] == ["path1", "path2"]
    assert mock_model_engine.call_args[1]["is_train"] is False
    mock_model.push_to_hub.assert_called_with("hub_id")
