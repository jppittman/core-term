#!/usr/bin/env bash
# Gate 1b: per program, node count, KernelKey::bits() and program_digest of what
# optimize_runtime_arena returns. usage: gate1b.sh TAG
# TAG=base records out/gate1b.base.txt; otherwise out/gate1b.TAG.txt must equal it.
set -u -o pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
check_tag "${1:-}"
TAG=$1
RES=$OUT/gate1b.$TAG.txt
guard_base "$TAG" "$OUT/gate1b.base.txt"

build_probe gate1b

RAW=$OUT/gate1b.$TAG.raw
ERR=$OUT/gate1b.$TAG.err
"$BIN" --ignored --nocapture --test-threads=1 zz_gate_font_kernels >"$RAW" 2>"$ERR" \
    || die "the probe failed; see $ERR"
probe_lines "$RAW" >"$OUT/gate1b.$TAG.lines"
# sorted by n, numerically: key each line by its n, sort on that, drop the key
awk '{ split($2, f, "="); print f[2] "\t" $0 }' "$OUT/gate1b.$TAG.lines" | sort -n -k1,1 | cut -f2- >"$RES"

[[ $(wc -l <"$RES") -eq 29 ]] || die "expected 29 ZZGATE lines, got $(wc -l <"$RES") (see $RAW)"
[[ $(cut -d' ' -f2 "$RES" | sort -u | wc -l) -eq 29 ]] || die "duplicate programs in $RES"

python3 "$GATES/gate1b_selfcheck.py" "$RES" "$REPO/$TESTS_RS" >&2 \
    || die "SELF-CHECK FAILED: the probe does not reproduce the suite's pinned digests"

echo "wrote $RES ($(wc -l <"$RES") lines, sha256 $(sha256sum "$RES" | cut -c1-16)...)"
[[ "$TAG" == base ]] && { echo "recorded as base"; exit 0; }
compare_to_base "$RES" "$OUT/gate1b.base.txt"
