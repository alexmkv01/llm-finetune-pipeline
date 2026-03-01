#!/bin/sh
set -x -e

echo "Running pipeline"

# Pull data and run cache from DVC remote
dvc pull --run-cache --allow-missing --force

{
    dvc repro
} || {
    printf "Saving partial output and artifacts\n"
    dvc push
    git add dvc.lock artifacts/
    git commit -m "dvc pipeline failed to complete [skip ci]"
    git push
    exit 1
}

echo "Saving reproduced output and artifacts"
dvc push

# Commit pipeline outputs
git add dvc.lock artifacts/
git commit --allow-empty -m "dvc pipeline reproduced [skip ci]"
git push

# Find associated PR for current branch
PR_NUMBER=$(gh pr list --head "$GITHUB_REF_NAME" --json number --jq '.[0].number')

if [ -z "$PR_NUMBER" ]; then
    echo "No PR found for branch $GITHUB_REF_NAME, skipping report"
    exit 0
fi

# Generate report using Jinja2 template
uv run llm-finetune-train training-report metrics_report.md

# Post report as PR comment
gh pr comment "$PR_NUMBER" --body-file metrics_report.md
