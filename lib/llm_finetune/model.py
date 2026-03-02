"""Model loading with transparent LoRA adapter detection and merging."""

import json
import logging
from pathlib import Path

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_finetune.exceptions import AdapterConfigError, CheckpointNotFoundError

logger = logging.getLogger(__name__)

ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def _is_lora_checkpoint(checkpoint_dir: Path) -> bool:
    """Check whether a checkpoint directory contains a LoRA adapter."""
    return (checkpoint_dir / ADAPTER_CONFIG_FILENAME).exists()


def _read_base_model_name(checkpoint_dir: Path) -> str:
    """Read the base model name from a LoRA adapter config.

    Raises:
        AdapterConfigError: If the config file is missing, unreadable, or
            does not contain a non-empty ``base_model_name_or_path``.
    """
    config_path = checkpoint_dir / ADAPTER_CONFIG_FILENAME
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise AdapterConfigError(
            f"failed to read adapter config at {config_path}: {e}"
        ) from e

    base_model_name = config.get("base_model_name_or_path")
    if not isinstance(base_model_name, str) or not base_model_name:
        raise AdapterConfigError(
            f"adapter config at {config_path} missing 'base_model_name_or_path'"
        )
    return base_model_name


def _load_lora_model(checkpoint_dir: Path) -> PreTrainedModel:
    """Load a LoRA adapter checkpoint and merge it into the base model."""
    base_model_name = _read_base_model_name(checkpoint_dir)
    logger.info(
        "Loading LoRA adapter from %s (base: %s)", checkpoint_dir, base_model_name
    )

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    # str() cast: transformers from_pretrained does not accept Path objects
    peft_model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    merged: PreTrainedModel = peft_model.merge_and_unload()
    logger.info("LoRA adapter merged successfully")
    return merged


def _load_full_model(checkpoint_dir: Path) -> PreTrainedModel:
    """Load a full (non-adapter) model checkpoint."""
    logger.info("Loading full model from %s", checkpoint_dir)
    # str() cast: transformers from_pretrained does not accept Path objects
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
    )
    return model


def load_model(
    checkpoint_dir: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model and tokenizer from a local HuggingFace checkpoint.

    Transparently handles both full checkpoints and LoRA adapter checkpoints.
    LoRA adapters are merged into the base model before returning.

    Raises:
        CheckpointNotFoundError: If the checkpoint directory does not exist.
        AdapterConfigError: If a LoRA adapter config is invalid.
    """
    if not checkpoint_dir.is_dir():
        raise CheckpointNotFoundError(
            f"checkpoint directory does not exist: {checkpoint_dir}"
        )

    if _is_lora_checkpoint(checkpoint_dir):
        model = _load_lora_model(checkpoint_dir)
    else:
        model = _load_full_model(checkpoint_dir)

    # str() cast: transformers from_pretrained does not accept Path objects
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
    )

    return model, tokenizer
