#!/bin/sh
# New code, MPL-2.0. Builds and runs the CPU-only timing accounting test into a
# fresh temporary directory. It compiles no D3D11 code and needs no GPU. The test
# runs once per inherited CHANDRA_NATIVE_CPU_TIMING state (absent, empty, accepted,
# rejected) and must restore each. The MSVC route is in docs/directcompute-cpu-timing.md.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/cpu-timing-test.XXXXXX")
trap 'rm -rf "$out"' EXIT
"${CXX:-c++}" -std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  -fsanitize=address,undefined -fno-sanitize-recover=all \
  "$here/cpu_timing_test.cpp" -o "$out/cpu-timing-test"
env -u CHANDRA_NATIVE_CPU_TIMING "$out/cpu-timing-test"
for inherited in "" 1 yes; do
  CHANDRA_NATIVE_CPU_TIMING="$inherited" "$out/cpu-timing-test"
done
