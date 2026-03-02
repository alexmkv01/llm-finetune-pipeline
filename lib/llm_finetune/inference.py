"""Text generation inference using a fine-tuned causal language model."""

import logging
from pathlib import Path

import torch

from llm_finetune.model import load_model
from llm_finetune.schemas import GenerationConfig

logger = logging.getLogger(__name__)


class SqlGenerationPipeline:
    """Pipeline wrapping a causal LM for natural-language-to-SQL generation.

    Loads a model checkpoint (full or LoRA) on construction and exposes
    a ``__call__`` interface for generating SQL from formatted prompt strings.

    Args:
        checkpoint_dir: Path to the model checkpoint directory. Loaded eagerly
            on construction — must exist and be readable.
        config: Generation parameters.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        config: GenerationConfig,
    ) -> None:
        self.config = config
        self.model, self.tokenizer = load_model(checkpoint_dir)
        logger.info("SqlGenerationPipeline initialised from %s", checkpoint_dir)

    def __call__(self, prompt: str) -> str:
        """Generate SQL from an already-formatted prompt string.

        The caller is responsible for prompt formatting. This method
        handles tokenisation, generation, and decoding only.

        Args:
            prompt: The fully formatted input prompt.

        Returns:
            The generated SQL string (without the input prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        input_length = input_ids.size(1)

        with torch.no_grad():
            # torch.nn.Module.__getattr__ stub types attribute access as
            # Tensor | Module, shadowing the real generate() method signature
            # and making mypy think we're calling a Tensor. This is a torch
            # stub limitation, not a real type error.
            output_ids = self.model.generate(  # type: ignore[operator]
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
            )

        generated_ids = output_ids[0, input_length:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise RuntimeError(
                f"tokenizer.decode returned {type(decoded).__name__}, expected str"
            )
        return decoded.strip()
