#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <cl-title> <base-commit> <head-commit>" >&2
  exit 2
fi

cl_title=$1
base_commit=$2
head_commit=$3
subject_pattern='^(fix|feat|chore|test|revert|docs|perf|refactor|ci)(\([[:alnum:]_.\/-]+\))?:[[:space:]]+[^[:space:]]'
expected='(fix|feat|chore|test|revert|docs|perf|refactor|ci)(scope)?: description'

check_subject() {
  local kind=$1
  local subject=$2

  if [[ $subject =~ $subject_pattern ]]; then
    return
  fi

  echo "invalid $kind: $subject" >&2
  echo "expected: $expected" >&2
  return 1
}

check_subject "CL title" "$cl_title"

git cat-file -e "${base_commit}^{commit}" || {
  echo "base is not a commit: $base_commit" >&2
  exit 2
}
git cat-file -e "${head_commit}^{commit}" || {
  echo "head is not a commit: $head_commit" >&2
  exit 2
}

commit_list=$(mktemp)
trap 'rm -f "$commit_list"' EXIT
# `--no-merges`: a merge commit records no change of its own, so it has no
# description to give — the subject is git's own generated text. Requiring a
# conventional subject there would reject every "Update branch", and main's
# own history already contains such merges.
git log --no-merges --format='%H%x09%s' "${base_commit}..${head_commit}" >"$commit_list" || {
  echo "failed to enumerate CL commits: ${base_commit}..${head_commit}" >&2
  exit 2
}

if [[ ! -s $commit_list ]]; then
  echo "CL contains no commits: ${base_commit}..${head_commit}" >&2
  exit 1
fi

invalid=0
while IFS=$'\t' read -r commit subject; do
  if ! check_subject "commit $commit" "$subject"; then
    invalid=1
  fi
done <"$commit_list"

if [[ $invalid -ne 0 ]]; then
  exit 1
fi

echo "CL title and all commit subjects satisfy: $expected"
