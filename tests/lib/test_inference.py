"""Tests for SqlGenerationPipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_finetune.exceptions import CheckpointNotFoundError
from llm_finetune.inference import SqlGenerationPipeline
from llm_finetune.schemas import GenerationConfig


class TestSqlGenerationPipeline:
    """SqlGenerationPipeline loads a model and generates SQL via __call__."""

    @patch("llm_finetune.inference.load_model")
    def test_init_loads_model_and_stores_config(
        self,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        config = GenerationConfig()

        pipeline = SqlGenerationPipeline(tmp_path, config)

        mock_load_model.assert_called_once_with(tmp_path)
        assert pipeline.model is mock_model
        assert pipeline.tokenizer is mock_tokenizer
        assert pipeline.config is config

    @patch("llm_finetune.inference.load_model")
    def test_init_propagates_checkpoint_not_found(
        self,
        mock_load_model: MagicMock,
    ) -> None:
        mock_load_model.side_effect = CheckpointNotFoundError("no such dir")
        with pytest.raises(CheckpointNotFoundError, match="no such dir"):
            SqlGenerationPipeline(Path("/nonexistent"), GenerationConfig())

    @patch("llm_finetune.inference.load_model")
    def test_call_generates_and_decodes(
        self,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Tokenizer returns input_ids and attention_mask
        mock_input_ids = MagicMock()
        mock_input_ids.to.return_value = mock_input_ids
        mock_input_ids.size = MagicMock(side_effect=lambda dim: 5 if dim == 1 else None)
        mock_attention_mask = MagicMock()
        mock_attention_mask.to.return_value = mock_attention_mask
        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        # model.generate returns output_ids
        mock_output_ids = MagicMock()
        mock_generated_slice = MagicMock()
        mock_output_ids.__getitem__ = MagicMock(return_value=mock_generated_slice)
        mock_model.generate.return_value = mock_output_ids

        # tokenizer.decode returns the SQL string
        mock_tokenizer.decode.return_value = " SELECT * FROM users "

        config = GenerationConfig(max_new_tokens=128, temperature=0.0, do_sample=False)
        pipeline = SqlGenerationPipeline(tmp_path, config)
        result = pipeline("Generate SQL for all users")

        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once_with(
            mock_generated_slice, skip_special_tokens=True
        )
        assert result == "SELECT * FROM users"
