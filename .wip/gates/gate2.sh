#!/usr/bin/env bash
# Gate 2: per program, the saturation's applications, unions, classes at stop,
# live classes, iterations and stop reason. usage: gate2.sh TAG
# The probe is built with pixelflow-core/saturation-telemetry and run
# single-threaded with the sink pointed at out/gate2.TAG.jsonl (truncated first,
# never deleted); the k-th record is the k-th program's.
# TAG=base records out/gate2.base.{txt,jsonl}; otherwise gate2.TAG.txt must equal gate2.base.txt.
set -u -o pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
check_tag "${1:-}"
TAG=$1
RES=$OUT/gate2.$TAG.txt
JSONL=$OUT/gate2.$TAG.jsonl
guard_base "$TAG" "$OUT/gate2.base.txt" "$OUT/gate2.base.jsonl"

build_probe gate2 --features pixelflow-core/saturation-telemetry

: >"$JSONL"
RAW=$OUT/gate2.$TAG.raw
ERR=$OUT/gate2.$TAG.err
PIXELFLOW_SATURATION_TELEMETRY=$JSONL "$BIN" --ignored --nocapture --test-threads=1 zz_gate_font_kernels \
    >"$RAW" 2>"$ERR" || die "the probe failed; see $ERR"
probe_lines "$RAW" >"$OUT/gate2.$TAG.lines"

python3 "$GATES/gate2_pair.py" "$OUT/gate2.$TAG.lines" "$JSONL" >"$RES" || die "pairing failed (see $JSONL, $RAW)"
[[ $(wc -l <"$RES") -eq 29 ]] || die "expected 29 lines in $RES, got $(wc -l <"$RES")"

echo "wrote $RES ($(wc -l <"$RES") lines, sha256 $(sha256sum "$RES" | cut -c1-16)...) and $JSONL ($(wc -l <"$JSONL") records)"
[[ "$TAG" == base ]] && { echo "recorded as base"; exit 0; }
compare_to_base "$RES" "$OUT/gate2.base.txt"
