#!/usr/bin/env bash
#
# `provenance-journal` (pixelflow-search/Cargo.toml) must never reach a
# downstream user's build: docs/plans/2026-09-01-production-budget-determinism.md
# gates the rule-provenance journal (origins, application log, union
# journal, `derivation_ancestors`, the hindsight labeler) behind this
# feature, default OFF, because a production compile built and discarded a
# median 8,446-application log per kernel for no consumer.
#
# Feature unification makes `cargo build --workspace` carry the feature into
# every crate that depends on pixelflow-search, once pixelflow-pipeline
# requests it -- that is expected and not what this checks. What must hold
# is the *downstream* dependency graph: a `-p`-scoped build of a crate that
# does not itself depend on pixelflow-pipeline must never see the feature,
# because that is the build a real `kernel!` consumer performs. `cargo tree`
# resolves each `-p` invocation independently, so these two checks stand in
# for "cargo build -p pixelflow-compiler" / "cargo build -p core-term" from
# a downstream user's own workspace.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail=0
for pkg in pixelflow-compiler core-term; do
  hit="$(cargo tree -p "$pkg" -e normal -f "{p} {f}" 2>&1 | grep provenance-journal || true)"
  if [[ -n "$hit" ]]; then
    echo "FAIL: provenance-journal reaches $pkg's normal dependency graph:" >&2
    echo "$hit" >&2
    fail=1
  else
    echo "OK: provenance-journal does not reach $pkg"
  fi
done

exit "$fail"
