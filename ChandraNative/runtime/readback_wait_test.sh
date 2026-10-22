#!/bin/sh
# New code, MPL-2.0. Builds and runs the CPU-only readback readiness control test
# into a fresh temporary directory. It compiles no D3D11 code and needs no GPU.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/readback-wait-test.XXXXXX")
trap 'rm -rf "$out"' EXIT
"${CXX:-c++}" -std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  -fsanitize=address,undefined -fno-sanitize-recover=all \
  "$here/readback_wait_test.cpp" -o "$out/readback-wait-test"
"$out/readback-wait-test"
