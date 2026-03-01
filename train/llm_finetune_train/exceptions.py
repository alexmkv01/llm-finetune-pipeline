"""Custom exceptions for llm-finetune-train."""


class LlmFineTuneTrainError(Exception):
    """Base exception for training pipeline errors."""


class DvcCommandError(LlmFineTuneTrainError):
    """Raised when a DVC subprocess command fails."""
