#!/usr/bin/env bash
#
# Every `src/bin/*.rs` must have an explicit `[[bin]]` entry in its crate's
# Cargo.toml.
#
# Cargo auto-discovers `src/bin/*.rs`, and an auto-discovered target carries
# no `required-features`. So a research bin that reads a feature-gated module
# -- `pixelflow-pipeline`'s `training` is the whole population of them --
# compiles fine for the author, who has the default features on, and breaks
# only under `--no-default-features`. That is the Feature matrix job, ~90s
# in, after the workspace has been checked 40 times.
#
# The declaration is the thing actually missing in that failure, and it is a
# property of two files sitting next to each other, so it costs a directory
# listing and a grep rather than a build. This is that check, shifted left.
#
# It deliberately does not require `required-features` on every bin --
# `collapse_cost` correctly has none. What it requires is that somebody wrote
# the entry down, which is the moment the question gets asked.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail=0
checked=0

while IFS= read -r manifest; do
  crate_dir="$(dirname "$manifest")"
  bin_dir="$crate_dir/src/bin"
  [[ -d "$bin_dir" ]] || continue

  while IFS= read -r bin_src; do
    rel="${bin_src#"$crate_dir"/}"
    checked=$((checked + 1))
    # `path = "src/bin/foo.rs"` is how every declared bin in this workspace
    # spells it; matching the path rather than the name keeps a stray
    # `name = "foo"` under some other target from counting as a declaration.
    if ! grep -qF "path = \"$rel\"" "$manifest"; then
      echo "FAIL: $bin_src has no [[bin]] entry in $manifest" >&2
      echo "      Auto-discovered bins carry no required-features; if this one" >&2
      echo "      reads a feature-gated module it will break the Feature matrix." >&2
      echo "      Add:" >&2
      echo "" >&2
      echo "        [[bin]]" >&2
      echo "        name = \"$(basename "$bin_src" .rs)\"" >&2
      echo "        path = \"$rel\"" >&2
      echo "        # required-features = [...]  # if it needs any" >&2
      echo "" >&2
      fail=1
    fi
  done < <(find "$bin_dir" -maxdepth 1 -name '*.rs' | sort)
done < <(find . -name Cargo.toml -not -path './target*' | sort)

if [[ "$fail" -eq 0 ]]; then
  echo "OK: all $checked src/bin targets are declared in their Cargo.toml"
fi

exit "$fail"
