#!/bin/sh
# New code, MPL-2.0. Builds and runs the CPU-only tied-head audit and native-1 mechanism test into a
# fresh temporary directory. It compiles no D3D11 code, needs no GPU and loads no model weights.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/head-audit-test.XXXXXX")
trap 'rm -rf "$out"' EXIT
"${CXX:-c++}" -std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Wshadow -Werror -ffp-contract=off \
  -fsanitize=address,undefined -fno-sanitize-recover=all \
  "$here/head_audit_test.cpp" "$here/operators.cpp" "$here/diagnostics.cpp" -o "$out/head-audit-test"
"$out/head-audit-test"
