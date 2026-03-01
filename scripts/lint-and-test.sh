#!/bin/sh
set -x -e

# ruff
ruff check .
ruff format --check .

# cspell - lint markdown templates
npx cspell@9 lint --no-progress "**/*.md.j2"

# mypy
mypy

# rumdl markdown linter
rumdl check --include "**/*.md.j2" .

# test
pytest
