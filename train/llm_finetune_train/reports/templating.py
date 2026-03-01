"""
Templating utilities for Jinja2 rendering of reports and LLM prompts.

Currently used for markdown report generation. The same module and naming
convention will serve LLM prompt templates when those are added.

Template Organisation:
    Templates are stored in the project root under templates/:
    - scripts/*.md.j2           - Report templates
    - {module}/prompts/*.md.j2  - LLM prompt templates (future)

Namespace Convention:
    Use dot notation: "module.template_name"

    Examples:
        "scripts.training_report"         -> scripts/training_report.md.j2
        "other_module.some_prompt"        -> other_module/prompts/some_prompt.md.j2

Usage:
    from llm_finetune_train.reports.templating import render_prompt

    output = render_prompt("scripts.training_report", metrics_table="...")
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TEMPLATES_ROOT = PROJECT_ROOT / "templates"

# Template naming constants
TEMPLATE_EXTENSION = ".md.j2"
NAMESPACE_SEPARATOR = "."
PROMPTS_SUBDIR = "prompts"
SCRIPTS_MODULE = "scripts"

# Jinja2 environment configuration
TEMPLATE_LOADER = FileSystemLoader(TEMPLATES_ROOT)
DEFAULT_ENV = Environment(
    loader=TEMPLATE_LOADER,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _resolve_template_path(template_id: str) -> str:
    """
    Resolve dot-notation identifier to template file path.

    Special case: scripts module templates are stored directly in scripts/
    All other modules store templates in {module}/prompts/ subdirectory.

    Args:
        template_id: Namespace identifier for the template.

    Returns:
        Relative path to template file from templates root.

    Examples:
        >>> _resolve_template_path("scripts.training_report")
        'scripts/training_report.md.j2'
        >>> _resolve_template_path("standalone_template")
        'standalone_template.md.j2'
    """
    if NAMESPACE_SEPARATOR not in template_id:
        return f"{template_id}{TEMPLATE_EXTENSION}"

    module, name = template_id.rsplit(NAMESPACE_SEPARATOR, 1)

    if module == SCRIPTS_MODULE:
        return f"{module}/{name}{TEMPLATE_EXTENSION}"

    return f"{module}/{PROMPTS_SUBDIR}/{name}{TEMPLATE_EXTENSION}"


def render_prompt(
    template_id: str,
    env: Environment = DEFAULT_ENV,
    **kwargs: object,
) -> str:
    """
    Render template using namespace identifier.

    Args:
        template_id: Dot-notation identifier for the template.
        env: Jinja2 environment to use for rendering.
        **kwargs: Variables to inject into template context.

    Returns:
        Rendered template string.

    Examples:
        >>> render_prompt("scripts.training_report", metrics_table="| col |")
        '# Training Report\\n\\n| col |...'
    """
    template_path = _resolve_template_path(template_id)
    template = env.get_template(template_path)
    return template.render(**kwargs)
