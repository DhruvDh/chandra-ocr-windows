#!/usr/bin/env bash
# New ChandraNative code, MPL-2.0. CPU-only B1 GEMV candidate checks: no GPU, model, network or
# install. Builds this checkout's operators.cpp and the immutable predecessor operators.cpp (commit
# 888ab33, read from Git) against the fake-device harness, then checks each selector state in its
# own process because CHANDRA_EXPERIMENTAL_GEMV_B1 is captured once at process start.
# Usage: ChandraNative/runtime/gemv_candidate_test.sh FRESH_BUILD_DIRECTORY
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Pass a fresh build directory" >&2; exit 2; }
build="$1"; mkdir "$build"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
predecessor=888ab33
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math -I"$here")
git -C "$here" show "$predecessor:ChandraNative/runtime/operators.cpp" > "$build/predecessor_operators.cpp"
g++ "${flags[@]}" "$here/gemv_candidate_test.cpp" "$here/operators.cpp" -o "$build/gemv-candidate-test"
g++ "${flags[@]}" -DGEMV_TEST_PREDECESSOR "$here/gemv_candidate_test.cpp" "$build/predecessor_operators.cpp" -o "$build/predecessor-test"

echo "== default route transcript equals predecessor $predecessor (unset and \"0\")"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/predecessor-test" --transcript > "$build/predecessor.txt"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/gemv-candidate-test" --transcript > "$build/unset.txt"
CHANDRA_EXPERIMENTAL_GEMV_B1=0 "$build/gemv-candidate-test" --transcript > "$build/zero.txt"
cmp "$build/predecessor.txt" "$build/unset.txt"
cmp "$build/predecessor.txt" "$build/zero.txt"
echo "identical: $(wc -l < "$build/predecessor.txt") transcript lines, sha256 $(sha256sum < "$build/predecessor.txt" | cut -c1-16)"

echo "== predecessor route under the new source"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/gemv-candidate-test" --predecessor
CHANDRA_EXPERIMENTAL_GEMV_B1=0 "$build/gemv-candidate-test" --predecessor

echo "== candidate route"
CHANDRA_EXPERIMENTAL_GEMV_B1=ordered "$build/gemv-candidate-test" --candidate
echo "batch>1 calls under the candidate selector keep the predecessor transcript:"
CHANDRA_EXPERIMENTAL_GEMV_B1=ordered "$build/gemv-candidate-test" --candidate-transcript > "$build/ordered.txt"
grep -E '^batch(2|33)_' "$build/predecessor.txt" > "$build/predecessor-batch.txt"
grep -E '^batch(2|33)_' "$build/ordered.txt" > "$build/ordered-batch.txt"
cmp "$build/predecessor-batch.txt" "$build/ordered-batch.txt"
echo "identical: $(wc -l < "$build/ordered-batch.txt") batch>1 transcript lines; B1 lines use runtime/linear_gemv.hlsl: $(grep -c 'linear_gemv' "$build/ordered.txt")"

echo "== invalid selector values are refused before allocation or dispatch"
for value in "" "1" "ORDERED" " ordered" "ordered " "true" "predecessor" "00"; do
  CHANDRA_EXPERIMENTAL_GEMV_B1="$value" "$build/gemv-candidate-test" --invalid
done

echo "== numerical observations (not acceptance thresholds)"
env -u CHANDRA_EXPERIMENTAL_GEMV_B1 "$build/gemv-candidate-test" --numerics > "$build/numerics.json"
cat "$build/numerics.json"
echo "ALL GEMV CANDIDATE CPU CHECKS PASSED"
