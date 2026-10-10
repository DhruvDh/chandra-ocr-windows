#!/usr/bin/env bash
# New ChandraNative code, MPL-2.0. CPU-only GEMM candidate checks: no GPU, D3D11, fxc, model,
# network or install. Confirms the production predecessor files are unchanged, builds
# gemm_candidate_fixture_test.cpp with strict GCC and Clang (full plan) and GCC ASan/UBSan
# (cases <= 50M terms), byte-compares their deterministic summaries and, when glslangValidator is
# already installed, parses the three shaders with its HLSL front end (not fxc; not authoritative).
# Usage: ChandraNative/runtime/gemm_candidate_fixture_test.sh FRESH_BUILD_DIRECTORY
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Pass a fresh build directory" >&2; exit 2; }
build="$1"; mkdir "$build"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
shaders="$repo/ChandraNative/shaders/runtime"
strict=(-std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror -ffp-contract=off -fno-fast-math)

echo "== production predecessor files: unchanged from HEAD and equal to the reviewed SHA-256"
production=(ChandraNative/shaders/runtime/linear.hlsl ChandraNative/runtime/operators.cpp ChandraNative/runtime/api.h
            ChandraNative/runtime/device.cpp ChandraNative/runtime/dispatch_calibration.cpp ChandraNative/runtime/shader_compile.cpp
            ChandraNative/runtime/build.cmd ChandraNative/build.cmd)
git -C "$repo" diff --quiet HEAD -- "${production[@]}"
# operators.cpp is 3e428af's: its ordered32/ordered64 GEMV selectors change only rows == 1 routing, and
# the multi-row linear.hlsl loop through end of file is byte-identical to da9e4b1's reviewed file.
(cd "$repo" && sha256sum -c --strict --quiet) <<'SUMS'
8fa9fb68f69b35a812f4a5821db11148e665611b0b5f560a5a0186ac13343d5f  ChandraNative/shaders/runtime/linear.hlsl
675fbad581aedc15e8448f28316006019095114298057121ac8f0252dd417ba2  ChandraNative/runtime/operators.cpp
563c7555da0b5c06c7a17cadc0722b8e252b7c9b867283dfe1a684e14bf5f9d1  ChandraNative/runtime/api.h
0fff6aa94cfe41f0738682718b1fedec5ef547a9fbfeab470114554e12e36d73  ChandraNative/runtime/device.cpp
b75a41910eadcba3f849254e10cfc4ec7835347d890056ec6b1524820de4ae85  ChandraNative/runtime/dispatch_calibration.cpp
76db610999723d36cff06c559390f3a58898f4b93eb192fe5ab601826b91e44c  ChandraNative/runtime/shader_compile.cpp
172d96841ffc55869b077772e3f28baea1d579afb48dede3bbb6be2cce5cd365  ChandraNative/runtime/build.cmd
40486964eb35f6f52bb442e197977ba73227b775862f7477438118c9560ee4fd  ChandraNative/build.cmd
SUMS
echo "unchanged: ${#production[@]} files"

echo "== strict builds"
g++ "${strict[@]}" -O2 "$here/gemm_candidate_fixture_test.cpp" -o "$build/gcc-test"
clang++ "${strict[@]}" -O2 "$here/gemm_candidate_fixture_test.cpp" -o "$build/clang-test"
g++ "${strict[@]}" -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all "$here/gemm_candidate_fixture_test.cpp" -o "$build/sanitized-test"

echo "== GCC, full plan"
"$build/gcc-test" "$shaders" "$build/gcc-summary.json"
echo "== Clang, full plan"
"$build/clang-test" "$shaders" "$build/clang-summary.json"
cmp "$build/gcc-summary.json" "$build/clang-summary.json"
echo "GCC and Clang oracle summaries are byte-identical: sha256 $(sha256sum < "$build/gcc-summary.json" | cut -c1-16)"
echo "== GCC ASan/UBSan, cases <= 50M terms"
"$build/sanitized-test" "$shaders" "$build/sanitized-summary.json" --skip-large

if command -v glslangValidator > /dev/null; then
  echo "== glslang HLSL front-end parse (secondary; root's fxc and the Intel driver are authoritative)"
  for name in linear linear_gemm_padded32 linear_gemm_padded64; do
    glslangValidator -D -e main -S comp -V -o "$build/$name.spv" "$shaders/$name.hlsl" > "$build/$name.glslang.txt" 2>&1
    if command -v spirv-val > /dev/null; then spirv-val "$build/$name.spv"; fi
    echo "$name.hlsl parsed"
  done
else
  echo "== glslangValidator absent: front-end parse skipped"
fi
echo "ALL GEMM CANDIDATE CPU CHECKS PASSED"
