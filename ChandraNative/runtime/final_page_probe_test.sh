#!/bin/sh
# New code, MPL-2.0. Builds and runs the CPU-only final-page probe tests into a fresh temporary directory.
# Optional $1: the authenticated 5120-byte source-row26213.bf16 to test admission and the splice. No D3D11, GPU or model.
# Also runs the source11 constant-upload negative control, which must be stopped by AddressSanitizer.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/final-page-probe-test.XXXXXX")
trap 'rm -rf "$out"' EXIT
"${CXX:-c++}" -std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Wshadow -Werror \
  -fsanitize=address,undefined -fno-sanitize-recover=all \
  "$here/final_page_probe_test.cpp" -o "$out/final-page-probe-test"
mkdir "$out/scratch"
"$out/final-page-probe-test" "$out/scratch" "$here/../shaders" "$@"
if ASAN_OPTIONS=detect_leaks=0 "$out/final-page-probe-test" --old-upload-overread >"$out/old.out" 2>"$out/old.err"; then
  echo "final_page_probe_test: source11 constant-upload overread was NOT detected" >&2; exit 1
fi
grep -q "ERROR: AddressSanitizer: stack-buffer-overflow" "$out/old.err" && grep -q "in source11Upload" "$out/old.err" || { echo "final_page_probe_test: negative control failed for another reason" >&2; cat "$out/old.err" >&2; exit 1; }
echo "final_page_probe_test: source11 32-byte upload into 256-byte constant buffer rejected by ASan (negative control)"
