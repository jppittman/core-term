#!/usr/bin/env bash
#
# Exhaustively check every crate against every one of its own features in
# isolation (cargo-hack's --each-feature), rather than relying on workspace
# `--all-features` runs. Cargo's feature unification means a workspace
# `--all-features` build can stay green while a crate's feature is broken on
# its own -- that's exactly how pixelflow-graphics's `serde` feature shipped
# broken: nothing else in the workspace needed `bitflags/serde`, so
# `--all-features` never noticed it wasn't wired up.
#
# No gap is currently known: KNOWN_BROKEN below is empty, and this script
# exists so any single-feature break gets caught automatically instead of
# requiring another one-off manual audit like the one that found the
# pixelflow-graphics bug in the first place.
# (pixelflow-ir's no_std build went green on 2026-08-02 and its `oracle`
# feature on 2026-08-08; pixelflow-search's std-off build joined them on
# 2026-08-30. All three are enforced by the `no-std` job now, so they are
# deliberately NOT listed below -- a regression must fail this script.)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Each entry is "<crate>|<space-separated --no-default-features feature args>"
# exactly as cargo-hack prints it in its "failed commands:" summary.
#
# Empty, and worth keeping that way: an entry here suppresses a real failure,
# so every one is a hole in this job. The two pixelflow-search rows that lived
# here were both the same defect -- a std-only weights loader used unguarded in
# `egraph/extraction.rs` -- and both the loader and the feature that gated it
# were deleted with the extraction-head program (2026-09-01). Nothing else is
# known broken, so any failure below is news.
KNOWN_BROKEN=()

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

set +e
cargo hack check --workspace --each-feature --no-dev-deps --keep-going >"$log" 2>&1
hack_status=$?
set -e

if [[ $hack_status -eq 0 ]]; then
  echo "OK: every crate's every feature builds in isolation"
  exit 0
fi

# Parse the "failed commands:" summary cargo-hack prints at the end:
#   failed commands:
#       <crate>:
#           `.../cargo check --manifest-path <crate>/Cargo.toml --no-default-features [--features <f>]`
mapfile -t failed_lines < <(
  awk '
    /^failed commands:/ { in_block = 1; next }
    !in_block { next }
    /^    [A-Za-z0-9_-]+:$/ { crate = $1; sub(/:$/, "", crate); next }
    /^        `/ {
      line = $0
      sub(/^        `/, "", line)
      sub(/`$/, "", line)
      print crate "|" line
    }
  ' "$log"
)

if [[ ${#failed_lines[@]} -eq 0 ]]; then
  echo "cargo-hack failed (exit $hack_status) but no 'failed commands:' summary was found -- treat as a hard failure" >&2
  tail -n 100 "$log" >&2
  exit 1
fi

unexpected=0
for entry in "${failed_lines[@]}"; do
  crate="${entry%%|*}"
  cmd="${entry#*|}"
  # Reduce the full invocation to "--no-default-features [--features X]",
  # dropping the cargo binary path and --manifest-path (both host-specific).
  features_part="$(echo "$cmd" | grep -oE -- '--no-default-features( --features [A-Za-z0-9_-]+)?' || true)"
  normalized="${features_part#--no-default-features}"
  normalized="${normalized# }"

  is_known=0
  for known in "${KNOWN_BROKEN[@]}"; do
    known_crate="${known%%|*}"
    known_features="${known#*|}"
    if [[ "$crate" == "$known_crate" && "$normalized" == "$known_features" ]]; then
      is_known=1
      break
    fi
  done

  if [[ $is_known -eq 1 ]]; then
    echo "known (tracked separately): $crate [$normalized]"
  else
    echo "UNEXPECTED FAILURE: $crate [$normalized] -- a feature broke in isolation that wasn't already known-broken" >&2
    unexpected=1
  fi
done

if [[ $unexpected -ne 0 ]]; then
  echo "" >&2
  echo "Full cargo-hack output:" >&2
  tail -n 200 "$log" >&2
  exit 1
fi

echo "OK: only the already-tracked known-broken combinations failed"
