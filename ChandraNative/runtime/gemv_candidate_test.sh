#!/usr/bin/env bash
# New ChandraNative code, MPL-2.0. CPU-only B1 GEMV candidate checks: no GPU, model, network or
# install. Builds this checkout's operators.cpp, the immutable predecessor operators.cpp (commit
# 888ab33) and the immutable ordered operators.cpp (commit 0935926), read from Git, against the
# fake-device harness, then checks each selector state in its own process because
# CHANDRA_EXPERIMENTAL_GEMV_B1 is captured once at process start. gemv_variants_test.cpp checks the
# selector/geometry helper, the staging index algebra and the variant HLSL source structure. Every
# build uses strict GCC warnings; Clang and AddressSanitizer/UndefinedBehaviorSanitizer builds repeat
# the helper and candidate checks. Single-threaded and deterministic.
# Usage: ChandraNative/runtime/gemv_candidate_test.sh FRESH_BUILD_DIRECTORY
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Pass a fresh build directory" >&2; exit 2; }
build="$1"; mkdir "$build"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shaders="$(cd "$here/../shaders/runtime" && pwd)"
predecessor=888ab33
ordered=0935926
routes=(ordered ordered32 ordered64)
flags=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror -ffp-contract=off -fno-fast-math -I"$here")
sanitize=(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)

echo "== immutable sources"
git -C "$here" diff --quiet "$ordered" -- "$shaders/linear_gemv.hlsl" "$shaders/linear.hlsl"
git -C "$here" diff --quiet "$predecessor" -- "$shaders/linear.hlsl"
echo "linear_gemv.hlsl and linear.hlsl equal commit $ordered; linear.hlsl equals $predecessor"
(cd "$shaders" && sha256sum linear.hlsl linear_gemv.hlsl linear_gemv_ordered32.hlsl linear_gemv_ordered64.hlsl)

git -C "$here" show "$predecessor:ChandraNative/runtime/operators.cpp" > "$build/predecessor_operators.cpp"
git -C "$here" show "$ordered:ChandraNative/runtime/operators.cpp" > "$build/ordered_operators.cpp"
g++ "${flags[@]}" "$here/gemv_candidate_test.cpp" "$here/operators.cpp" -o "$build/gemv-candidate-test"
g++ "${flags[@]}" -DGEMV_TEST_PREDECESSOR "$here/gemv_candidate_test.cpp" "$build/predecessor_operators.cpp" -o "$build/predecessor-test"
g++ "${flags[@]}" "$here/gemv_candidate_test.cpp" "$build/ordered_operators.cpp" -o "$build/ordered-reference-test"
g++ "${flags[@]}" "$here/gemv_variants_test.cpp" -o "$build/gemv-variants-test"
clang++ "${flags[@]}" "$here/gemv_variants_test.cpp" -o "$build/gemv-variants-test-clang"
clang++ "${flags[@]}" "$here/gemv_candidate_test.cpp" "$here/operators.cpp" -o "$build/gemv-candidate-test-clang"
g++ "${flags[@]}" "${sanitize[@]}" "$here/gemv_variants_test.cpp" -o "$build/gemv-variants-test-sanitized"
g++ "${flags[@]}" "${sanitize[@]}" "$here/gemv_candidate_test.cpp" "$here/operators.cpp" -o "$build/gemv-candidate-test-sanitized"

echo "== selector/geometry helper, staging invariants and HLSL source structure"
"$build/gemv-variants-test" "$shaders"
"$build/gemv-variants-test-clang" "$shaders" > "$build/helper-clang.txt"
"$build/gemv-variants-test-sanitized" "$shaders" > "$build/helper-sanitized.txt"
tail -1 "$build/helper-clang.txt"; tail -1 "$build/helper-sanitized.txt"
"$build/gemv-variants-test" "$shaders" --forecast > "$build/forecast.txt"
cat "$build/forecast.txt"

echo "== default route transcript equals predecessor $predecessor (unset and \"0\")"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/predecessor-test" --transcript > "$build/predecessor.txt"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/gemv-candidate-test" --transcript > "$build/unset.txt"
CHANDRA_EXPERIMENTAL_GEMV_B1=0 "$build/gemv-candidate-test" --transcript > "$build/zero.txt"
cmp "$build/predecessor.txt" "$build/unset.txt"
cmp "$build/predecessor.txt" "$build/zero.txt"
echo "identical: $(wc -l < "$build/predecessor.txt") transcript lines, sha256 $(sha256sum < "$build/predecessor.txt" | cut -c1-16)"

echo "== predecessor route under the new source"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/gemv-candidate-test" --predecessor
CHANDRA_EXPERIMENTAL_GEMV_B1=0 "$build/gemv-candidate-test" --predecessor > /dev/null
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/gemv-candidate-test-sanitized" --predecessor > /dev/null

echo "== ordered route transcript equals commit $ordered"
CHANDRA_EXPERIMENTAL_GEMV_B1=ordered "$build/ordered-reference-test" --candidate-transcript > "$build/ordered-reference.txt"
CHANDRA_EXPERIMENTAL_GEMV_B1=ordered "$build/gemv-candidate-test" --candidate-transcript > "$build/ordered.txt"
cmp "$build/ordered-reference.txt" "$build/ordered.txt"
echo "identical: $(wc -l < "$build/ordered.txt") transcript lines, sha256 $(sha256sum < "$build/ordered.txt" | cut -c1-16)"

grep -E '^batch(2|33)_' "$build/predecessor.txt" > "$build/predecessor-batch.txt"
grep -E ' output_fnv1a64=' "$build/predecessor.txt" > "$build/predecessor-outputs.txt"
for route in "${routes[@]}"; do
  echo "== candidate route $route"
  CHANDRA_EXPERIMENTAL_GEMV_B1="$route" "$build/gemv-candidate-test" --candidate
  CHANDRA_EXPERIMENTAL_GEMV_B1="$route" "$build/gemv-candidate-test-clang" --candidate > /dev/null
  CHANDRA_EXPERIMENTAL_GEMV_B1="$route" "$build/gemv-candidate-test-sanitized" --candidate > /dev/null
  echo "Clang and sanitized builds: PASS"
  [[ "$route" == ordered ]] || CHANDRA_EXPERIMENTAL_GEMV_B1="$route" "$build/gemv-candidate-test" --candidate-transcript > "$build/$route.txt"
  grep -E '^batch(2|33)_' "$build/$route.txt" > "$build/$route-batch.txt"
  cmp "$build/predecessor-batch.txt" "$build/$route-batch.txt"
  grep -E ' output_fnv1a64=' "$build/$route.txt" > "$build/$route-outputs.txt"
  cmp "$build/predecessor-outputs.txt" "$build/$route-outputs.txt"
  shader="runtime/linear_gemv.hlsl"; [[ "$route" == ordered ]] || shader="runtime/linear_gemv_$route.hlsl"
  echo "batch>1 lines equal the predecessor ($(wc -l < "$build/$route-batch.txt")); all $(wc -l < "$build/$route-outputs.txt") output hashes equal the predecessor; B1 dispatch lines use $shader: $(grep -c " $shader " "$build/$route.txt")"
done

echo "== invalid selector values are refused before allocation or dispatch"
for value in "" "1" "ORDERED" " ordered" "ordered " "true" "predecessor" "00" "ordered8" "ordered16" "Ordered32" "ordered32 " "ordered 64" "ordered640"; do
  CHANDRA_EXPERIMENTAL_GEMV_B1="$value" "$build/gemv-candidate-test" --invalid
done
CHANDRA_EXPERIMENTAL_GEMV_B1=ordered128 "$build/gemv-candidate-test-sanitized" --invalid > /dev/null

echo "== numerical observations (not acceptance thresholds)"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/gemv-candidate-test" --numerics > "$build/numerics.json"
cat "$build/numerics.json"
echo "ALL GEMV CANDIDATE CPU CHECKS PASSED"
