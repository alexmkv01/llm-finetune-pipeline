"""Tests for template path resolution in templating.py."""

import pytest

from llm_finetune_train.reports.templating import _resolve_template_path


@pytest.mark.parametrize(
    ("template_id", "expected"),
    [
        (
            "scripts.training_report",
            "scripts/training_report.md.j2",
        ),
        (
            "standalone_template",
            "standalone_template.md.j2",
        ),
        (
            "other_module.some_prompt",
            "other_module/prompts/some_prompt.md.j2",
        ),
    ],
)
def test_resolve_template_path(template_id: str, expected: str) -> None:
    """_resolve_template_path maps dot-notation IDs to correct file paths."""
    assert _resolve_template_path(template_id) == expected


def test_multiple_dots_uses_last_separator() -> None:
    """rsplit on last separator: 'a.b.c' -> module='a.b', name='c'."""
    result = _resolve_template_path("a.b.c")
    assert result == "a.b/prompts/c.md.j2"
