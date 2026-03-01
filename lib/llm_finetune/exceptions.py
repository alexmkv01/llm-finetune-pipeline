"""Custom exceptions for llm-finetune-pipeline."""


class LlmFinetuneError(Exception):
    """Base exception for llm-finetune-pipeline."""


class CheckpointNotFoundError(LlmFinetuneError):
    """Raised when a model checkpoint directory does not exist."""


class AdapterConfigError(LlmFinetuneError):
    """Raised when a LoRA adapter config is invalid or incompatible."""


class GenerationError(LlmFinetuneError):
    """Raised when text generation fails."""
