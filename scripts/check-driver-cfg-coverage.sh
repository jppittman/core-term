#!/usr/bin/env bash
#
# Every display-driver cfg a build.rs can select must be either built by some
# CI job (blocking) or explicitly tracked by a known-gap status job
# (non-blocking) -- never silently untested. This is the check that would
# have caught the headless driver: it was declared via rustc-check-cfg,
# selectable via DISPLAY_DRIVER=headless, had 22 compile errors, and no CI
# job ever built it, for years, with a fully green history.
#
# This script does not discover coverage automatically -- that's the point.
# COVERAGE below is a manually maintained ledger, and an *unmapped* cfg fails
# the build. Adding a new display driver without updating this file is a CI
# failure, not a silent gap.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mapfile -t build_scripts < <(find . -name build.rs -not -path "./target/*" -not -path "./target.noindex/*" | sort)
if [[ ${#build_scripts[@]} -eq 0 ]]; then
  echo "no build.rs files found" >&2
  exit 2
fi

mapfile -t declared < <(grep -ohE 'use_[a-z0-9_]+_display' "${build_scripts[@]}" | sort -u)
if [[ ${#declared[@]} -eq 0 ]]; then
  echo "no use_*_display cfgs declared -- check the grep pattern still matches build.rs" >&2
  exit 2
fi

declare -A COVERAGE=(
  [use_cocoa_display]="implicit: macos-latest test/clippy jobs resolve to cocoa by default (build.rs target_os priority)"
  [use_x11_display]="implicit: ubuntu-latest test/clippy jobs resolve to x11 by default (build.rs target_os priority)"
  [use_web_display]="explicit: wasm32-status job in rust.yaml (non-blocking, tracks a known pre-existing gap)"
)

missing=0
for cfg in "${declared[@]}"; do
  if [[ -n "${COVERAGE[$cfg]:-}" ]]; then
    echo "OK: $cfg -> ${COVERAGE[$cfg]}"
  else
    echo "::error::display driver cfg '$cfg' is declared in a build.rs but has no entry in COVERAGE (scripts/check-driver-cfg-coverage.sh). Add a CI job that builds it -- or an explicit non-blocking status job if it's a known gap -- and record how here." >&2
    missing=1
  fi
done

exit "$missing"
