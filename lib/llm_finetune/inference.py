"""Text generation inference using a fine-tuned causal language model."""

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm_finetune.exceptions import GenerationError
from llm_finetune.schemas import GenerationConfig

logger = logging.getLogger(__name__)


def generate_sql(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: GenerationConfig,
) -> str:
    """
    Generate SQL from an already-formatted prompt string.

    The caller is responsible for prompt formatting. This function handles
    tokenisation, generation, and decoding only.

    Args:
        model: The loaded causal language model.
        tokenizer: The tokenizer matching the model.
        prompt: The fully formatted input prompt.
        config: Generation parameters.

    Returns:
        The generated SQL string (without the input prompt).

    Raises:
        GenerationError: If tokenisation or generation fails.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        input_length = input_ids.size(1)

        with torch.no_grad():
            output_ids = model.generate(  # type: ignore[operator]
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
            )

        generated_ids = output_ids[0, input_length:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise GenerationError("expected single string from decode, got list")
        return decoded.strip()

    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"generation failed: {e}") from e
