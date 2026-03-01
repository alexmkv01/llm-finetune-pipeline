"""Tests for model loading and LoRA adapter detection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_finetune.exceptions import AdapterConfigError, CheckpointNotFoundError
from llm_finetune.model import _is_lora_checkpoint, _read_base_model_name, load_model


class TestIsLoraCheckpoint:
    """_is_lora_checkpoint detects adapter config presence."""

    def test_returns_true_when_adapter_config_exists(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("{}")
        assert _is_lora_checkpoint(tmp_path) is True

    def test_returns_false_when_no_adapter_config(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}")
        assert _is_lora_checkpoint(tmp_path) is False


class TestReadBaseModelName:
    """_read_base_model_name extracts the base model from adapter config."""

    def test_reads_base_model_name(self, tmp_path: Path) -> None:
        config = '{"base_model_name_or_path": "HuggingFaceTB/SmolLM2-1.7B"}'
        (tmp_path / "adapter_config.json").write_text(config)
        assert _read_base_model_name(tmp_path) == "HuggingFaceTB/SmolLM2-1.7B"

    def test_raises_on_missing_key(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("{}")
        with pytest.raises(AdapterConfigError, match="missing"):
            _read_base_model_name(tmp_path)

    def test_raises_on_malformed_json(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("not json")
        with pytest.raises(AdapterConfigError, match="failed to read"):
            _read_base_model_name(tmp_path)


class TestLoadModel:
    """load_model validates the checkpoint directory and dispatches correctly."""

    def test_raises_on_missing_directory(self) -> None:
        with pytest.raises(CheckpointNotFoundError, match="does not exist"):
            load_model(Path("/nonexistent/checkpoint"))

    @patch("llm_finetune.model.AutoTokenizer")
    @patch("llm_finetune.model.AutoModelForCausalLM")
    def test_loads_full_model_when_no_adapter(
        self,
        mock_auto_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "config.json").write_text("{}")
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model, tokenizer = load_model(tmp_path)

        mock_auto_model.from_pretrained.assert_called_once_with(str(tmp_path))
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(str(tmp_path))
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    @patch("llm_finetune.model.AutoTokenizer")
    @patch("llm_finetune.model.PeftModel")
    @patch("llm_finetune.model.AutoModelForCausalLM")
    def test_loads_and_merges_lora_adapter(
        self,
        mock_auto_model: MagicMock,
        mock_peft_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = '{"base_model_name_or_path": "HuggingFaceTB/SmolLM2-1.7B"}'
        (tmp_path / "adapter_config.json").write_text(config)

        mock_base = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_base
        mock_merged = MagicMock()
        mock_peft_instance = MagicMock()
        mock_peft_instance.merge_and_unload.return_value = mock_merged
        mock_peft_model.from_pretrained.return_value = mock_peft_instance
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model, tokenizer = load_model(tmp_path)

        mock_auto_model.from_pretrained.assert_called_once_with(
            "HuggingFaceTB/SmolLM2-1.7B"
        )
        mock_peft_model.from_pretrained.assert_called_once_with(
            mock_base, str(tmp_path)
        )
        mock_peft_instance.merge_and_unload.assert_called_once()
        assert model is mock_merged
        assert tokenizer is mock_tokenizer
