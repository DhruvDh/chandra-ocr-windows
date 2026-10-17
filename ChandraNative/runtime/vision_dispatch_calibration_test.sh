#!/usr/bin/env bash
# New ChandraNative code, MPL-2.0. CPU-only vision dispatch calibration checks: no GPU, model, weights,
# network or install. Builds the unit suite with g++ and clang++ (strict warnings as errors) plus a clang
# ASan/UBSan build, links the unchanged vision_model.cpp/operators.cpp, then runs the unchanged Windows
# entry vision_dispatch_calibration.cpp through a POSIX receipt stub on the fake Device. Single-threaded.
# Usage: ChandraNative/runtime/vision_dispatch_calibration_test.sh FRESH_BUILD_DIRECTORY
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Pass a fresh build directory" >&2; exit 2; }
build="$(realpath -m "$1")"; mkdir "$build"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; shaders="$here/../shaders"
own=(-std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror -ffp-contract=off -fno-fast-math -I"$here")
prod=(-std=c++17 -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math)

compile() { # compiler label flags...
  local cxx=$1 out="$build/$2"; shift 2; mkdir "$out"
  local quiet=(); [[ $cxx == g++ ]] && quiet=(-Wno-dangling-reference) # GCC false positive in unchanged vision_model.cpp.
  "$cxx" "${prod[@]}" "${quiet[@]}" "$@" -c "$here/vision_model.cpp" -o "$out/vision_model.o"
  "$cxx" "${prod[@]}" "$@" -c "$here/operators.cpp" -o "$out/operators.o"
  "$cxx" "${own[@]}" "$@" "$here/vision_dispatch_calibration_test.cpp" "$out/vision_model.o" "$out/operators.o" -o "$out/test"
  mkdir "$out/stub"; cp "$here/vision_dispatch_calibration_windows_stub.h" "$out/stub/windows.h"
  "$cxx" "${own[@]}" "$@" -I"$out/stub" "$here/vision_dispatch_calibration.cpp" "$here/vision_dispatch_calibration_entry_harness.cpp" -o "$out/entry"
}
peak() { python3 -I -c 'import resource,subprocess,sys
r=subprocess.run(sys.argv[1:])
print("peak resident KiB:", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
sys.exit(r.returncode)' "$@"; }

echo "== build: g++ -O2, clang++ -O2, clang++ -O1 ASan/UBSan"
compile g++ gcc -O2
compile clang++ clang -O2
compile clang++ sanitize -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all

echo "== unit suite (g++)"; mkdir "$build/gcc/scratch"; peak "$build/gcc/test" "$shaders" "$build/gcc/scratch"
echo "== unit suite (clang++)"; mkdir "$build/clang/scratch"; "$build/clang/test" "$shaders" "$build/clang/scratch" | tail -n 1
cmp <(python3 -I -c 'import json,sys; r=json.load(open(sys.argv[1])); [p.pop(k,None) for p in r["phases"] for k in ("upload_seconds","oracle_seconds","elapsed_seconds")]; [s.pop(k,None) for p in r["phases"] for s in p["stages"] for k in ("host_submit_to_drained_milliseconds","host_readback_milliseconds")]; print(json.dumps(r,sort_keys=True))' "$build/gcc/scratch/fake-full-report.json") \
    <(python3 -I -c 'import json,sys; r=json.load(open(sys.argv[1])); [p.pop(k,None) for p in r["phases"] for k in ("upload_seconds","oracle_seconds","elapsed_seconds")]; [s.pop(k,None) for p in r["phases"] for s in p["stages"] for k in ("host_submit_to_drained_milliseconds","host_readback_milliseconds")]; print(json.dumps(r,sort_keys=True))' "$build/clang/scratch/fake-full-report.json")
echo "g++ and clang++ fake receipts are identical apart from host wall times"
echo "== unit suite (clang++ ASan/UBSan, quick: phases 1-3 executed, CLI tile transcript only)"
mkdir "$build/sanitize/scratch"; "$build/sanitize/test" "$shaders" "$build/sanitize/scratch" --quick | tail -n 1

echo "== unchanged Windows entry on the fake device"
entry="$build/gcc/entry"; out="$build/receipts"; mkdir "$out"
expect() { # name expected-exit command...
  local name=$1 want=$2; shift 2; local got=0
  "$@" >"$out/$name.stdout" 2>"$out/$name.stderr" || got=$?
  [[ $got -eq $want ]] || { echo "FAIL $name: exit $got, wanted $want"; cat "$out/$name.stderr"; exit 1; }
  echo "  $name: exit $got $(head -c 160 "$out/$name.stderr" | tr '\n' ' ')"
}
receipt() { python3 -I - "$@" <<'EOF'
import json, sys
path, checks = sys.argv[1], sys.argv[2:]
r = json.load(open(path))
assert len(open(path, "rb").read()) <= 8 * 1024 * 1024
for check in checks:
    key, want = check.split("=", 1)
    value = r
    for part in key.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    if want.startswith("~"):
        assert want[1:] in str(value), (key, value)
    else:
        assert json.dumps(value) == want, (key, value, want)
print("    receipt", path.rsplit("/", 1)[-1], "ok:", " ".join(checks))
EOF
}
pci=03:00.0; luid=00000000:0000abcd
expect inactive 0 "$entry"
python3 -I -c 'import json,sys; r=json.loads(open(sys.argv[1]).read()); assert r["state"]=="INACTIVE" and r["device_created"] is False and r["plan"]["dispatches"]==16; print("    inactive plan:", len(open(sys.argv[1]).read()), "bytes, 16 planned dispatches, no device")' "$out/inactive.stdout"
[[ "$(sed -n 1p "$out/inactive.stderr")" == "fake dispatches 0" ]]
expect inactive-with-options 0 "$entry" --pci "$pci"
expect unknown-option 1 "$entry" --model x
expect duplicate-option 1 "$entry" --pci "$pci" --pci "$pci"
expect duplicate-execute 1 "$entry" --execute --execute
# Review VDC-03: a value position never swallows an option token, so no flag can vanish into a value; only
# genuinely valid arguments without --execute print the inactive plan.
refuse() { # name stderr-fragment arguments...
  local name=$1 want=$2; shift 2
  expect "$name" 1 "$entry" "$@"
  [[ ! -s "$out/$name.stdout" ]] || { echo "FAIL $name: refusal printed a plan"; exit 1; }
  grep -qF -- "$want" "$out/$name.stderr" || { echo "FAIL $name: stderr lacks: $want"; exit 1; }
  grep -qx "fake dispatches 0" "$out/$name.stderr" || { echo "FAIL $name: dispatched"; exit 1; }
}
refuse valueless-pci-consumes-execute "Option value absent or option-shaped after --pci" --pci --execute
refuse unknown-flag-consumed-as-pci "Option value absent or option-shaped after --pci" --pci --unknown
refuse execute-pci-swallows-luid "after --pci" --execute --shader-root "$shaders" --pci --luid "$luid" --output "$out/swallow.json"
refuse execute-root-swallows-pci "after --shader-root" --execute --shader-root --pci "$pci" --luid "$luid" --output "$out/swallow.json"
refuse output-swallows-execute "after --output" --shader-root "$shaders" --pci "$pci" --luid "$luid" --output --execute
refuse empty-pci "after --pci" --pci ""
refuse dash-luid "after --luid" --luid -
refuse trailing-option "after --luid" --pci "$pci" --luid
refuse unknown-after-valid "Unknown calibration option" --pci "$pci" --unknown
refuse execute-with-unknown "Unknown calibration option" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/swallow.json" --unknown
refuse stray-value "Unknown calibration option" --pci "$pci" extra
refuse equals-form "Unknown calibration option" --pci="$pci"
refuse duplicate-luid "Duplicate calibration option --luid" --luid "$luid" --luid "$luid"
refuse duplicate-execute-with-options "Duplicate execute flag" --pci "$pci" --execute --execute
refuse malformed-pci-inactive "PCI pin must be lowercase" --pci 3:00.0
refuse uppercase-luid-inactive "LUID pin must be lowercase" --luid 00000000:0000ABCD
refuse relative-root-inactive "Absolute shader root and output paths required" --shader-root ChandraNative/shaders
[[ ! -e "$out/swallow.json" ]]
expect inactive-all-valid 0 "$entry" --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/never.json"
python3 -I -c 'import json,sys; r=json.loads(open(sys.argv[1]).read()); assert r["state"]=="INACTIVE" and r["device_created"] is False' "$out/inactive-all-valid.stdout"
[[ ! -e "$out/never.json" && "$(sed -n 1p "$out/inactive-all-valid.stderr")" == "fake dispatches 0" ]]
expect missing-luid 1 "$entry" --execute --shader-root "$shaders" --pci "$pci" --output "$out/missing-luid.json"
expect uppercase-luid 1 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid 00000000:0000ABCD --output "$out/uppercase.json"
expect relative-output 1 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output relative.json
expect relative-shaders 1 "$entry" --execute --shader-root ChandraNative/shaders --pci "$pci" --luid "$luid" --output "$out/relative-shaders.json"
for f in missing-luid uppercase relative-shaders; do [[ ! -e "$out/$f.json" ]]; done
echo "precious" > "$out/existing.json"
expect existing-output 1 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/existing.json"
[[ "$(cat "$out/existing.json")" == "precious" ]]
mkdir "$out/empty-shaders"
expect unfrozen-shaders 1 "$entry" --execute --shader-root "$out/empty-shaders" --pci "$pci" --luid "$luid" --output "$out/unfrozen.json"
receipt "$out/unfrozen.json" state='"FAIL_OR_INCOMPLETE"' device_created=false native_executed=false retirement_required=false error='~Frozen production shader absent' dispatches_submitted=0
expect softmax-fault 1 env VDC_FAKE_FAULT=softmaxDropLastLane "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/softmax-fault.json"
receipt "$out/softmax-fault.json" state='"FAIL_OR_INCOMPLETE"' native_executed=true retirement_required=true dispatches_submitted=2 \
    failure_drain.completed=true failure_drain.tracked_after=0 phases.0.stages.1.check.passed=false phases.0.stages.2.dispatch_call_started=false
expect gate-at-dispatch-6 1 env VDC_FAKE_SLOW_DISPATCH=6 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/gate.json"
receipt "$out/gate.json" dispatches_submitted=6 dispatches_completed_and_measured=6 phases.1.stages.1.strictly_below_gate=false \
    phases.1.stages.1.gpu_milliseconds=100.0 error='~at or above the 100 ms gate' failure_drain.tracked_after=0
expect disjoint-at-dispatch-1 1 env VDC_FAKE_PROFILE_FAULT=disjointThrows VDC_FAKE_PROFILE_FAULT_AT=1 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/disjoint.json"
receipt "$out/disjoint.json" dispatches_submitted=1 dispatches_completed_and_measured=0 phases.0.stages.0.submitted=true phases.0.stages.0.completed=false error='~disjoint'
# Review VDC-02: a receiptCap+1-byte driver message is stored as a bounded excerpt with length and hash, and
# giant nested profile metadata yields only the bounded RECEIPT_OVERSIZE summary and stops further dispatch.
expect giant-driver-error 1 env VDC_FAKE_PROFILE_FAULT=giantError VDC_FAKE_PROFILE_FAULT_AT=2 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/giant-error.json"
receipt "$out/giant-error.json" state='"FAIL_OR_INCOMPLETE"' retirement_required=true dispatches_submitted=2 dispatches_completed_and_measured=1 \
    phases.0.stages.1.submitted=true phases.0.stages.1.completed=false phases.0.stages.2.dispatch_call_started=false \
    error='~[ERROR TRUNCATED, detail lost: 8388637 bytes, sha256 ' failure_drain.completed=true failure_drain.tracked_after=0
[[ $(stat -c %s "$out/giant-error.json") -lt 1048576 && $(stat -c %s "$out/giant-driver-error.stderr") -lt 70000 ]]
expect giant-profile-metadata 1 env VDC_FAKE_PROFILE_FAULT=giantMetadata VDC_FAKE_PROFILE_FAULT_AT=3 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/giant-metadata.json"
receipt "$out/giant-metadata.json" state='"RECEIPT_OVERSIZE"' passed=false detail_lost=true retirement_required=true native_executed=true \
    dispatches_submitted=3 dispatches_completed_and_measured=3 original_state.excerpt='"FAIL_OR_INCOMPLETE"' \
    original_error.excerpt='~Receipt exceeded the 8 MiB cap; detail lost; no further dispatch' failure_drain.completed=true \
    failure_drain.tracked_after=0 largest_top_level_fields.0.key='"phases"'
[[ $(stat -c %s "$out/giant-metadata.json") -lt 65536 && "$(tail -n 1 "$out/giant-profile-metadata.stderr")" == "fake dispatches 3" ]]
expect pass 0 "$entry" --execute --shader-root "$shaders" --pci "$pci" --luid "$luid" --output "$out/pass.json"
receipt "$out/pass.json" state='"VISION_DISPATCH_CALIBRATION_PASS"' passed=true native_executed=true dispatches_submitted=16 \
    dispatches_completed_and_measured=16 memory_after.tracked_live=0 phases.3.sequence_length=12096 phases.3.stages.0.groups='[95, 512, 1]' \
    phases.3.stages.1.check.uniform.passed=true cli_attention_projection.query_tiles_per_block=378.0 full_model_accepted=false \
    frozen_shader_verification.4.sha256='"442da3bfcebe567fdb1f5f97b392557ceba0b9f71505435012dc615e565988c5"'
echo "ALL VISION DISPATCH CALIBRATION CPU CHECKS PASSED"
