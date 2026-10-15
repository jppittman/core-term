#!/usr/bin/env bash
# Every module directory carries an intent.md, shaped to the template in
# docs/plans/2026-10-15-intent-lives-with-the-module.md, naming only files the
# tree has. The logic is in check_intent.py; this is the CI entry point.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 scripts/check_intent.py
