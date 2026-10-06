#!/usr/bin/env bash
# New ChandraNative code, MPL-2.0. CPU-only checks of the text softmax multiwave regression: no GPU, D3D11,
# fxc, model, weights, network or install. Builds text_softmax_regression_test.cpp (core header plus the
# test-only fake Device and host) with g++ and clang++ under strict warnings-as-errors and a clang++
# ASan/UBSan build, runs each, and requires byte-identical g++/clang++ summaries. The Windows host
# text_softmax_regression.cpp is MSVC-only; root compiles it. The translated-HLSL emulation harness runs from
# tests/native/test_exact_input_correctness.py. Single-threaded.
# Usage: ChandraNative/runtime/text_softmax_regression_test.sh FRESH_BUILD_DIRECTORY
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Pass a fresh build directory" >&2; exit 2; }
build="$(realpath -m "$1")"; mkdir "$build"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; shaders="$(cd "$here/../shaders" && pwd)"
own=(-std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror -ffp-contract=off -fno-fast-math -I"$here")
compile() { # compiler label flags...
  local cxx=$1 out="$build/$2"; shift 2; mkdir "$out"
  "$cxx" "${own[@]}" "$@" "$here/text_softmax_regression_test.cpp" -o "$out/test" 2> "$out/compiler.log" || { cat "$out/compiler.log"; exit 1; }
  [[ ! -s "$out/compiler.log" ]] || { echo "$cxx emitted diagnostics:"; cat "$out/compiler.log"; exit 1; }
  echo "  $cxx $* : no diagnostics"
}
echo "== build: g++ -O2, clang++ -O2, clang++ -O1 ASan/UBSan"
compile g++ gcc -O2
compile clang++ clang -O2
compile clang++ sanitize -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all
for b in gcc clang; do
  echo "== unit suite ($b)"; mkdir "$build/$b/scratch"
  "$build/$b/test" "$shaders" "$build/$b/scratch" > "$build/$b/summary.json"
  tail -n 1 "$build/$b/summary.json" | head -c 160; echo
done
cmp "$build/gcc/summary.json" "$build/clang/summary.json"
echo "g++ and clang++ summaries are byte-identical"
echo "== unit suite (clang++ ASan/UBSan, quick: one wave model, no entry lifecycle matrix)"
mkdir "$build/sanitize/scratch"; "$build/sanitize/test" "$shaders" "$build/sanitize/scratch" --quick > "$build/sanitize/summary.json"
tail -n 1 "$build/sanitize/summary.json" | head -c 160; echo
echo "ALL TEXT SOFTMAX REGRESSION CPU CHECKS PASSED"
