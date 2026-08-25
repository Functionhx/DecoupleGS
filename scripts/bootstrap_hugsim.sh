#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_dir="$root_dir/third_party/HUGSIM"
patch_file="$root_dir/patches/decouplegs.patch.gz"
base_commit="62c690d39fd90020e68a196bd8bcc1c4d4191f2e"

if [[ ! -d "$base_dir/.git" && ! -f "$base_dir/.git" ]]; then
  git -C "$root_dir" submodule update --init --recursive
fi

actual_commit="$(git -C "$base_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$base_commit" ]]; then
  echo "HUGSIM is at $actual_commit; expected $base_commit." >&2
  echo "Run: git -C third_party/HUGSIM checkout --detach $base_commit" >&2
  exit 1
fi

if gzip -cd "$patch_file" | git -C "$base_dir" apply --check --binary; then
  gzip -cd "$patch_file" | git -C "$base_dir" apply --binary
  echo "Applied the DecoupleGS overlay to third_party/HUGSIM."
else
  echo "The overlay cannot be applied cleanly. It may already be applied, or the submodule was modified." >&2
  exit 1
fi
