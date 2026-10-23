#!/usr/bin/env bash
# Gate 2x (after gate2.sh TAG): the extraction half of the same telemetry records.
# usage: gate2x.sh TAG
# TAG=base records out/gate2x.base.txt from gate2.base.jsonl; otherwise
# gate2x_fields.py on gate2.TAG.jsonl must equal it.
set -u -o pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
check_tag "${1:-}"
TAG=$1
RES=$OUT/gate2x.$TAG.txt
[[ -s "$OUT/gate2.$TAG.jsonl" ]] || die "no $OUT/gate2.$TAG.jsonl; run gate2.sh $TAG first"
guard_base "$TAG" "$OUT/gate2x.base.txt"

python3 "$GATES/gate2x_fields.py" "$OUT/gate2.$TAG.jsonl" >"$RES" || die "could not read $OUT/gate2.$TAG.jsonl"
[[ $(wc -l <"$RES") -eq 29 ]] || die "expected 29 records, got $(wc -l <"$RES")"
echo "wrote $RES ($(wc -l <"$RES") lines, sha256 $(sha256sum "$RES" | cut -c1-16)...)"
[[ "$TAG" == base ]] && { echo "recorded as base"; exit 0; }
compare_to_base "$RES" "$OUT/gate2x.base.txt"
