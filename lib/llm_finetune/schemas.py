"""Pydantic models and type definitions for llm-finetune-pipeline."""

from pydantic import BaseModel, field_validator


class GenerationConfig(BaseModel):
    """Parameters controlling text generation behaviour.

    Args:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. 0 = greedy, higher = more random.
        top_p: Nucleus sampling probability mass cutoff.
        repetition_penalty: Penalty for repeating tokens. 1.0 = no penalty.
        do_sample: Whether to use sampling. False = greedy decoding.
    """

    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    do_sample: bool = True

    @field_validator("max_new_tokens")
    @classmethod
    def max_new_tokens_must_be_positive(cls, v: int) -> int:
        """Must generate at least one token."""
        if v <= 0:
            raise ValueError("max_new_tokens must be > 0")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_non_negative(cls, v: float) -> float:
        """Temperature must be >= 0."""
        if v < 0:
            raise ValueError("temperature must be >= 0")
        return v

    @field_validator("top_p")
    @classmethod
    def top_p_must_be_in_unit_interval(cls, v: float) -> float:
        """Nucleus sampling cutoff must be in (0, 1]."""
        if v <= 0 or v > 1:
            raise ValueError("top_p must be in (0, 1]")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def repetition_penalty_must_be_positive(cls, v: float) -> float:
        """Repetition penalty must be > 0."""
        if v <= 0:
            raise ValueError("repetition_penalty must be > 0")
        return v
