"""CLI entry point for llm-finetune-train."""

import logging
import subprocess
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg

from llm_finetune_train.exceptions import DvcCommandError
from llm_finetune_train.reports.templating import render_prompt

logger = logging.getLogger(__name__)


class TrainingReport(BaseSettings):
    """Generate training metrics report from DVC metrics."""

    output_file: CliPositionalArg[Path] = Field(
        description="Output markdown file path",
    )
    base_branch: str = Field(default="main", description="Base branch for metrics diff")

    def cli_cmd(self) -> None:
        """Generate and write training report."""
        logger.info("Generating training report against %s", self.base_branch)

        metrics_table = _fetch_metrics_diff(self.base_branch)

        report = render_prompt(
            "scripts.training_report",
            metrics_table=metrics_table,
        )
        self.output_file.write_text(report)
        logger.info("Report written to %s", self.output_file)


def _fetch_metrics_diff(base_branch: str) -> str:
    """
    Run dvc metrics diff and return markdown output.

    Args:
        base_branch: Base branch for metrics comparison.

    Returns:
        Markdown table from DVC metrics diff.

    Raises:
        DvcCommandError: If the dvc command fails.
    """
    result = subprocess.run(
        ["dvc", "metrics", "diff", base_branch, "--md"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DvcCommandError(result.stderr.strip())
    return result.stdout.strip()


_COMMANDS: dict[str, type[BaseSettings]] = {
    "training-report": TrainingReport,
}


def main() -> None:
    """CLI entry point."""
    _, *args = sys.argv

    if not args or args[0] not in _COMMANDS:
        available = ", ".join(_COMMANDS)
        print(
            f"Usage: llm-finetune-train <command> [options]\n"
            f"Available commands: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    CliApp.run(_COMMANDS[args[0]], cli_args=args[1:])


if __name__ == "__main__":
    main()
