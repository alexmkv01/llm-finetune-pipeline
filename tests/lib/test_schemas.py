"""Tests for GenerationConfig validation."""

import pytest
from pydantic import ValidationError

from llm_finetune.schemas import GenerationConfig


class TestGenerationConfigDefaults:
    """GenerationConfig default values are sensible for NL-to-SQL."""

    def test_default_max_new_tokens(self) -> None:
        config = GenerationConfig()
        assert config.max_new_tokens == 256

    def test_default_temperature(self) -> None:
        config = GenerationConfig()
        assert config.temperature == 0.1

    def test_default_top_p(self) -> None:
        config = GenerationConfig()
        assert config.top_p == 0.95

    def test_default_repetition_penalty(self) -> None:
        config = GenerationConfig()
        assert config.repetition_penalty == 1.1

    def test_default_do_sample(self) -> None:
        config = GenerationConfig()
        assert config.do_sample is True


class TestGenerationConfigValidation:
    """GenerationConfig validates input parameters."""

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValidationError, match="temperature must be >= 0"):
            GenerationConfig(temperature=-0.1)

    def test_zero_temperature_allowed(self) -> None:
        config = GenerationConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_zero_max_new_tokens_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_new_tokens must be > 0"):
            GenerationConfig(max_new_tokens=0)

    def test_negative_max_new_tokens_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_new_tokens must be > 0"):
            GenerationConfig(max_new_tokens=-10)

    def test_zero_top_p_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"top_p must be in \(0, 1\]"):
            GenerationConfig(top_p=0.0)

    def test_top_p_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"top_p must be in \(0, 1\]"):
            GenerationConfig(top_p=1.01)

    def test_top_p_one_allowed(self) -> None:
        config = GenerationConfig(top_p=1.0)
        assert config.top_p == 1.0

    def test_zero_repetition_penalty_raises(self) -> None:
        with pytest.raises(ValidationError, match="repetition_penalty must be > 0"):
            GenerationConfig(repetition_penalty=0.0)

    def test_negative_repetition_penalty_raises(self) -> None:
        with pytest.raises(ValidationError, match="repetition_penalty must be > 0"):
            GenerationConfig(repetition_penalty=-1.0)

    def test_override_all_fields(self) -> None:
        config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=False,
        )
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.2
        assert config.do_sample is False

    def test_serialisation_round_trip(self) -> None:
        original = GenerationConfig(max_new_tokens=128, temperature=0.5)
        dumped = original.model_dump()
        restored = GenerationConfig(**dumped)
        assert original == restored
