#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
result_root_input=${1:-$script_dir}
result_root=$(cd -- "$result_root_input" && pwd -P)
manifest="$result_root/RESULTS-MANIFEST.tsv"
manifest_tmp=$(mktemp "$result_root/.results-manifest.XXXXXX")
compressed_tmp=

cleanup() {
  if [[ -n "$compressed_tmp" && -e "$compressed_tmp" ]]; then
    rm -f -- "$compressed_tmp"
  fi
  if [[ -e "$manifest_tmp" ]]; then
    rm -f -- "$manifest_tmp"
  fi
}
trap cleanup EXIT

for command_name in find sort gzip sha256sum stat mktemp mv awk; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$command_name" >&2
    exit 1
  fi
done

printf 'raw_path\traw_bytes\traw_sha256\tgzip_path\tgzip_bytes\tgzip_sha256\n' \
  >"$manifest_tmp"

count=0
while IFS= read -r -d '' raw_log; do
  relative_path=${raw_log#"$result_root"/}
  if [[ "$relative_path" == *$'\t'* || "$relative_path" == *$'\n'* ]]; then
    printf 'unsupported tab or newline in path: %q\n' "$relative_path" >&2
    exit 1
  fi

  compressed_log="${raw_log}.gz"
  compressed_tmp="${compressed_log}.tmp"
  before=$(stat -c '%s:%y' -- "$raw_log")

  gzip -n -1 -c -- "$raw_log" >"$compressed_tmp"
  gzip -t -- "$compressed_tmp"

  after=$(stat -c '%s:%y' -- "$raw_log")
  if [[ "$before" != "$after" ]]; then
    printf 'log changed while being archived: %s\n' "$relative_path" >&2
    exit 1
  fi

  raw_sha256=$(sha256sum -- "$raw_log" | awk '{print $1}')
  archived_raw_sha256=$(gzip -dc -- "$compressed_tmp" | sha256sum | awk '{print $1}')
  if [[ "$raw_sha256" != "$archived_raw_sha256" ]]; then
    printf 'compressed content mismatch: %s\n' "$relative_path" >&2
    exit 1
  fi

  gzip_bytes=$(stat -c '%s' -- "$compressed_tmp")
  gzip_sha256=$(sha256sum -- "$compressed_tmp" | awk '{print $1}')
  raw_bytes=${before%%:*}

  mv -f -- "$compressed_tmp" "$compressed_log"
  compressed_tmp=

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$relative_path" \
    "$raw_bytes" \
    "$raw_sha256" \
    "${relative_path}.gz" \
    "$gzip_bytes" \
    "$gzip_sha256" \
    >>"$manifest_tmp"

  ((count += 1))
  if ((count % 100 == 0)); then
    printf 'archived %d logs\n' "$count"
  fi
done < <(
  find "$result_root" -type f -name '*.log' \
    -not -path '*/bin/*' \
    -not -path "$result_root/ptxas_shim_log/*" \
    -print0 |
    sort -z
)

mv -f -- "$manifest_tmp" "$manifest"
printf 'archived %d logs; manifest: %s\n' "$count" "$manifest"
