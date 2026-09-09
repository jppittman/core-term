#!/usr/bin/env bash
# The beam-extraction sweep: `Greedy` against `Beam::width(k)` at two class
# caps, on the benchmark correction's DEV families. Deterministic columns
# only (`--no-clock --no-probe`) — `dag_cost` is a property of the term, so
# every number here is exact and the shared host's load cannot move it.
#
# Usage: scripts/beam-extraction-sweep.sh <out-dir> "<widths>" "<caps>" "<filters>"
set -euo pipefail

OUT="${1:?out dir}"
WIDTHS="${2:-1 4 16 64}"
CAPS="${3:-5000 50000}"
FILTERS="${4:-shader: cellgrid psychedelic}"

BIN=./target.noindex/release/egraph_off_on
mkdir -p "$OUT"

for cap in $CAPS; do
  for k in $WIDTHS; do
    for f in $FILTERS; do
      tag="cap${cap}-beam${k}"
      echo "=== $tag $f (load: $(uptime | sed 's/.*load/load/')) ==="
      PIXELFLOW_GUARD_TELEMETRY=1 "$BIN" run \
        --out "$OUT/$tag.jsonl" \
        --filter "$f" \
        --class-cap "$cap" \
        --beam "$k" \
        --no-clock --no-probe
    done
  done
  # The `Greedy` control at the same cap: no `--beam` at all, so the
  # production extractor runs through the identical path.
  for f in $FILTERS; do
    echo "=== cap${cap}-greedy $f ==="
    PIXELFLOW_GUARD_TELEMETRY=1 "$BIN" run \
      --out "$OUT/cap${cap}-greedy.jsonl" \
      --filter "$f" \
      --class-cap "$cap" \
      --no-clock --no-probe
  done
done
