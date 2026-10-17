"""Compare a native diagnostic dump with a retained reference dump (standard library only).

Dumps are directories written by ChandraNative/runtime/diagnostics.cpp or by write_dump() below:
progress.jsonl plus one raw little-endian payload per record. Every payload is re-authenticated by
size and SHA-256, every declared observation count and BF16 boundary is recomputed from the payload,
every text record's consumed-prefix commitment is recomputed from the dump's own conditioning rows, and
the terminal line must agree with the authenticated plan and the parsed progress. Incompatible
model/input commitments and malformed, contradictory or incomplete dumps are refused rather than compared.

Records pair by stage, layer/block, phase, decode step, state cache length and selected row. A pair is
comparable only when its shape, decode metadata, semantic coordinates and complete consumed prefix agree;
the producer's call/tile structure and completed cache length are provenance and never part of the join.

Tolerances are explicit caller arguments; without them metrics are reported with no verdict. Agreement
with a CPU oracle is operator/graph evidence only. Nothing here establishes primary AMD 7900 XTX OCR
qualification, complete-page acceptance or native throughput.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

PROGRESS_SCHEMA = "chandra.directcompute.diagnostic-progress.v2"
RECORD_SCHEMA = "chandra.directcompute.diagnostic-record.v2"
PREFIX_SCHEMA = "chandra.directcompute.consumed-prefix.v1"
NATIVE_PLAN_SCHEMA = "chandra.directcompute.diagnostic-plan.v1"
REFERENCE_PLAN_SCHEMA = "chandra.directcompute.diagnostic-reference-plan.v2"
COMPARISON_SCHEMA = "chandra.directcompute.diagnostic-comparison.v2"
BF16_POLICY = "bf16_round_to_nearest_even_at_graph_boundary"
STATE_POLICY = "none_fp32_state"
VOCABULARY = 248320
NAME = re.compile(r"[a-z0-9_][a-z0-9._-]{0,159}")
DIGEST = re.compile(r"[0-9a-f]{64}")
PROGRESS_LIMIT = 64 * 1024 * 1024
PAYLOAD_LIMIT = 256 * 1024 * 1024
TOTAL_LIMIT = 2 * 1024 * 1024 * 1024
RECORD_LIMIT = 65536
ITEM = {"float32": ("f", 4), "float64": ("d", 8)}
WORD = "I" if array("I").itemsize == 4 else "L"
# Text values conditioned on a causal prefix of consumed request rows. Row stages bind the prefix
# through their absolute row; carried GDN state binds the prefix ending at its cache length.
ROW_STAGES = ("text.layer_output", "text.final_norm", "text.logits")
STATE_STAGES = ("text.gdn_conv_state", "text.gdn_recurrent_state")
CONDITIONED = ROW_STAGES + STATE_STAGES
# Coordinates that identify the same graph value independently of producer tiling or call structure.
# Record "cache" (completed cache length, call rows) and coordinates call_row/chunk_index/chunk_row are
# producer provenance only and are never compared.
SEMANTIC = ("absolute_row", "token_id", "position", "source", "merged_row", "prompt_row", "patch_row",
            "frame", "grid_row", "grid_col", "last_absolute_row", "last_position")
IDENTITY = ("stage", "layer", "vision_block", "phase", "decode_step", "cache_length", "logical_shape",
            "selected_axis", "selected_rows", "payload_shape", "complete_tensor", "produces_generated_index", "decode")
CRITERIA = ("max_abs", "max_rel", "max_rmse", "min_cosine", "max_bf16_ulp", "require_argmax_match")
TERMINAL = {"event", "run_status", "dump_complete", "planned_records", "written_records", "missing_records",
            "incomplete_record", "payload_bytes", "error", "consumed_rows", "consumed_prefix_sha256", "qualified"}


class Refusal(Exception):
    """The inputs cannot be compared honestly."""


def require(condition, message):
    if not condition:
        raise Refusal(message)


def strict_json(text, where):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{where}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise Refusal(f"{where}: nonfinite JSON constant {value}")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as error:
        raise Refusal(f"{where}: invalid JSON: {error}") from None
    require(isinstance(value, dict), f"{where}: JSON object required")
    return value


def natural(value):
    return type(value) is int and value >= 0


def optional_natural(value):
    return value is None or natural(value)


def canonical_sha256(value):
    """SHA-256 of compact, key-sorted UTF-8 JSON: the encoding nlohmann::json::dump() gives the native plan."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key_text(stage, index, phase, step, row, cache):
    part = lambda v: "-" if v is None else str(v)
    return f"{stage}/i{part(index)}/{phase}/s{part(step)}/r{part(row)}/c{part(cache)}"


def record_key(record):
    """The native recorder's record identity, derived from the record's own identity fields."""
    index = record.get("layer") if record.get("layer") is not None else record.get("vision_block")
    selected = record.get("selected_rows")
    return key_text(record["stage"], index, record["phase"], record.get("decode_step"),
                    None if selected == "all" else selected[0], record.get("cache_length"))


class History:
    """Consumed request rows [token_id, t, h, w] of one dump and the SHA-256 of every prefix.

    The canonical stream is "<PREFIX_SCHEMA>\\ninput_manifest_sha256 <hex>\\nprompt_rows <P>\\n" followed by
    "<absolute_row> <token_id> <t> <h> <w>\\n" per consumed row; digests[n] covers the header and n rows.
    """

    def __init__(self, manifest, prompt_rows):
        self.prompt_rows = prompt_rows
        self.rows = []
        self.hash = hashlib.sha256(f"{PREFIX_SCHEMA}\ninput_manifest_sha256 {manifest}\nprompt_rows {prompt_rows}\n".encode())
        self.digests = [self.hash.hexdigest()]

    def append(self, row):
        require(isinstance(row, list) and len(row) == 4 and all(natural(v) for v in row) and row[0] < VOCABULARY,
                f"consumed row {len(self.rows)} must be [token_id, t, h, w] naturals within the vocabulary")
        self.hash.update(f"{len(self.rows)} {row[0]} {row[1]} {row[2]} {row[3]}\n".encode())
        self.rows.append(list(row))
        self.digests.append(self.hash.hexdigest())


def observe(record, data):
    """Nonfinite and finite-but-not-exactly-BF16 element counts of a raw payload."""
    if record["storage_dtype"] == "float32":
        words = array(WORD)
        words.frombytes(data)
        if sys.byteorder != "little":
            words.byteswap()
        nonfinite = sum(1 for w in words if (w & 0x7F800000) == 0x7F800000)
        other = sum(1 for w in words if (w & 0x7F800000) != 0x7F800000 and w & 0xFFFF)
    else:
        doubles = array("d")
        doubles.frombytes(data)
        if sys.byteorder != "little":
            doubles.byteswap()
        finite = [math.isfinite(v) for v in doubles]
        singles = array("f", (v if f else 0.0 for v, f in zip(doubles, finite)))  # Out-of-range values become inf.
        words = array(WORD, singles.tobytes())
        nonfinite = finite.count(False)
        other = sum(1 for v, f, s, w in zip(doubles, finite, singles, words) if f and (s != v or w & 0xFFFF))
    return {"nonfinite": nonfinite, "finite_non_bf16_representable": other}


def regular_file(path, limit, what):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise Refusal(f"{what} is missing: {path.name}") from None
    require(stat.S_ISREG(info.st_mode), f"{what} must be a regular non-symlink file: {path.name}")
    require(info.st_size <= limit, f"{what} exceeds its {limit}-byte bound: {path.name}")
    return info.st_size


def check_record(record, header, model, where):
    for field in ("sequence", "file", "key", "stage", "phase", "logical_shape", "selected_rows", "payload_shape",
                  "complete_tensor", "storage_dtype", "byte_order", "bf16_rounding", "payload_bytes", "payload_sha256",
                  "coordinates", "commitments", "producer", "qualified", "observed", "conditioning"):
        require(field in record, f"{where}: record field {field} missing")
    require(record.get("schema") == RECORD_SCHEMA, f"{where}: record schema differs")
    require(record["qualified"] is False, f"{where}: a raw diagnostic record cannot claim qualification")
    name = record["file"]
    require(isinstance(name, str) and NAME.fullmatch(name) and ".." not in name and name != "progress.jsonl",
            f"{where}: unsafe payload filename {name!r}")
    require(record["storage_dtype"] in ITEM and record["byte_order"] == "little", f"{where}: unsupported payload storage")
    require(record["bf16_rounding"] in (BF16_POLICY, STATE_POLICY, "none"), f"{where}: unknown rounding policy")
    require(isinstance(record["stage"], str) and isinstance(record["phase"], str), f"{where}: stage and phase must be strings")
    for field in ("layer", "vision_block", "decode_step", "cache_length"):
        require(optional_natural(record.get(field)), f"{where}: {field} must be null or a nonnegative integer")
    text = record["stage"].startswith("text.")
    require(record.get("vision_block") is None if text else record.get("layer") is None,
            f"{where}: text records name a layer and vision records a block, never both")
    logical, payload = record["logical_shape"], record["payload_shape"]
    require(isinstance(logical, list) and logical and all(natural(v) and v > 0 for v in logical),
            f"{where}: logical shape must be positive integers")
    selected = record["selected_rows"]
    if selected == "all":
        require(payload == logical and record["complete_tensor"] is True, f"{where}: complete selection shape differs")
    else:
        require(isinstance(selected, list) and len(selected) == 1 and natural(selected[0]) and selected[0] < logical[0],
                f"{where}: selected rows must be one logical row")
        require(payload == [1] + logical[1:], f"{where}: payload shape does not equal selected rows")
        require(record["complete_tensor"] is (logical[0] == 1), f"{where}: complete_tensor flag is inconsistent")
    require(isinstance(record["coordinates"], list) and len(record["coordinates"]) == 1
            and isinstance(record["coordinates"][0], dict), f"{where}: one coordinate object per record required")
    elements = math.prod(payload)
    require(record["payload_bytes"] == elements * ITEM[record["storage_dtype"]][1] and record["payload_bytes"] <= PAYLOAD_LIMIT,
            f"{where}: payload byte count differs from its shape")
    require(isinstance(record["payload_sha256"], str) and DIGEST.fullmatch(record["payload_sha256"]),
            f"{where}: lowercase payload SHA-256 required")
    seen = record["observed"]
    require(isinstance(seen, dict) and set(seen) == {"nonfinite", "finite_non_bf16_representable"}
            and all(natural(v) for v in seen.values()), f"{where}: observed counts must be two nonnegative integers")
    expected = dict(header["commitments"])
    expected.update({k: model[k] for k in ("model_sha256", "config_sha256", "model_bytes") if k in model})
    require(record["commitments"] == expected, f"{where}: record commitments differ from the dump commitments")
    require(record["producer"] == header["producer"], f"{where}: record producer differs from the dump producer")


def check_conditioning(record, history, where):
    """A text record must commit to a prefix of this dump's consumed rows that agrees with its own metadata."""
    commitment, stage = record["conditioning"], record["stage"]
    if stage not in CONDITIONED:
        require(commitment is None, f"{where}: only causal text records carry a consumed-prefix commitment")
        return
    require(history is not None and isinstance(commitment, dict) and set(commitment) == {"prefix_rows", "prefix_sha256"},
            f"{where}: text record lacks its consumed-prefix commitment")
    n = commitment["prefix_rows"]
    require(natural(n) and 1 <= n <= len(history.rows),
            f"{where}: consumed prefix is not covered by the preceding conditioning rows")
    require(commitment["prefix_sha256"] == history.digests[n], f"{where}: consumed-prefix SHA-256 differs from the dump's conditioning rows")
    prompt, row, coordinate = history.prompt_rows, history.rows[n - 1], record["coordinates"][0]
    decode = n > prompt
    step = n - prompt - 1 if decode else None
    require(record["phase"] == ("decode" if decode else "prefill") and record.get("decode_step") == step,
            f"{where}: phase or decode step differs from its consumed prefix")
    if decode:
        require(record.get("decode") == {"step": step, "consumed_generated_index": step, "consumed_token_id": row[0],
                                         "logits_produce_generated_index": step + 1},
                f"{where}: decode metadata differs from the consumed row it names")
    else:
        require(record.get("decode") is None, f"{where}: prefill record carries decode metadata")
    if stage in STATE_STAGES:
        require(record.get("cache_length") == n and coordinate.get("last_absolute_row") == n - 1
                and coordinate.get("last_position") == row[1:], f"{where}: state endpoint differs from its consumed prefix")
        return
    require(record.get("cache_length") is None and coordinate.get("absolute_row") == n - 1
            and coordinate.get("token_id") == row[0] and coordinate.get("position") == row[1:],
            f"{where}: causal row, token or position differs from its consumed prefix")
    if stage == "text.logits":
        require(record["selected_rows"] == [0] and (decode or n == prompt)
                and record.get("produces_generated_index") == (step + 1 if decode else 0),
                f"{where}: logits row differs from the last consumed row it produces from")
    elif decode:
        require(record["selected_rows"] == [0] and record["logical_shape"][0] == 1, f"{where}: decode rows are [1, width]")
    else:
        require(record["selected_rows"] == [n - 1] and record["logical_shape"][0] == prompt,
                f"{where}: prefill rows are absolute rows of the [prompt, width] tensor")


def extend_history(history, line, where):
    require(history is not None, f"{where}: conditioning rows without a declared consumed-prefix schema")
    require(set(line) == {"event", "first_row", "rows", "consumed_rows", "prefix_sha256"}, f"{where}: conditioning fields differ")
    rows = line["rows"]
    require(line["first_row"] == len(history.rows) and isinstance(rows, list) and rows,
            f"{where}: conditioning rows must continue the consumed prefix contiguously")
    for row in rows:
        history.append(row)
    require(line["consumed_rows"] == len(history.rows) and line["prefix_sha256"] == history.digests[-1],
            f"{where}: conditioning prefix SHA-256 differs from its rows")


def native_plan_keys(plan, prompt):
    """Record identities that the native resolver enumerates for a resolved plan."""
    keys = []

    def rows(value):
        require(isinstance(value, list) and all(natural(v) for v in value), "native plan selections must be index lists")
        return value

    def flag(value):
        require(type(value) is bool, "native plan switches must be booleans")
        return value

    try:
        vision, prefill, decode = plan["vision"], plan["prefill"], plan["decode"]
        for enabled, stage in ((vision["patch_embedding"], "vision.patch_embedding"), (vision["position_added"], "vision.position_added")):
            if flag(enabled):
                keys += [key_text(stage, None, "vision", None, r, None) for r in rows(vision["rows"])]
        for block in rows(vision["blocks"]):
            keys += [key_text("vision.block_output", block, "vision", None, r, None) for r in rows(vision["rows"])]
        keys += [key_text("vision.merger_output", None, "vision", None, r, None) for r in rows(plan["merger"]["rows"])]
        keys += [key_text("text.merged_embedding", None, "merge", None, r, None) for r in rows(plan["embedding"]["rows"])]
        for layer in rows(prefill["layers"]):
            keys += [key_text("text.layer_output", layer, "prefill", None, r, None) for r in rows(prefill["rows"])]
        if flag(prefill["final_norm"]):
            keys += [key_text("text.final_norm", None, "prefill", None, r, None) for r in rows(prefill["rows"])]
        if flag(prefill["logits"]):
            keys.append(key_text("text.logits", None, "prefill", None, 0, None))
        for step in rows(decode["steps"]):
            keys += [key_text("text.layer_output", layer, "decode", step, 0, None) for layer in rows(decode["layers"])]
            if flag(decode["final_norm"]):
                keys.append(key_text("text.final_norm", None, "decode", step, 0, None))
            if flag(decode["logits"]):
                keys.append(key_text("text.logits", None, "decode", step, 0, None))
        require(isinstance(plan["gdn_state"], list), "native plan gdn_state must be a list")
        for state in plan["gdn_state"]:
            require(natural(state["layer"]), "native plan state layers must be indices")
            for length in rows(state["cache_lengths"]):
                require(prompt is not None, "native plan state lengths require the declared prompt length")
                phase, step = ("decode", length - prompt - 1) if length > prompt else ("prefill", None)
                if flag(state["conv"]):
                    keys.append(key_text("text.gdn_conv_state", state["layer"], phase, step, None, length))
                keys += [key_text("text.gdn_recurrent_state", state["layer"], phase, step, h, length)
                         for h in rows(state["recurrent_heads"])]
    except (KeyError, TypeError, AttributeError):
        raise Refusal("native plan selections are malformed") from None
    require(len(keys) == len(set(keys)), "native plan selections repeat a record identity")
    return sorted(keys)


def authenticate_plan(header, prompt):
    """Verify the plan commitment and return its exact planned record identities and forecast."""
    plan, digest = header.get("plan"), header.get("plan_sha256")
    require(isinstance(plan, dict), "The plan line must carry the resolved plan object")
    require(isinstance(digest, str) and digest == canonical_sha256(plan), "plan_sha256 differs from the canonical resolved plan")
    require(header["commitments"].get("plan_sha256") == digest, "Dump commitments name a different plan_sha256")
    forecast = plan.get("forecast")
    require(isinstance(forecast, dict) and natural(forecast.get("records")) and isinstance(forecast.get("keys"), list),
            "The plan forecast must state its record count and keys")
    keys = forecast["keys"]
    require(keys and all(isinstance(k, str) for k in keys) and keys == sorted(set(keys)) and len(keys) == forecast["records"],
            "Plan forecast keys must be sorted, unique and equal to its record count")
    if plan.get("schema") == NATIVE_PLAN_SCHEMA:
        require(native_plan_keys(plan, prompt) == keys, "Native plan forecast keys differ from its resolved selections")
    else:
        require(plan.get("schema") == REFERENCE_PLAN_SCHEMA, f"Unknown plan schema {plan.get('schema')!r}")
    return set(keys), forecast, plan


def check_terminal(finished, planned, forecast, plan, records, started, payload_total, history):
    """Validate the finished line against the plan and parsed progress; return whether the dump is complete."""
    fields = set(finished)
    require(TERMINAL <= fields and fields <= TERMINAL | {"readbacks", "readback_bytes"}, "Finished line fields differ from the terminal contract")
    status, claim, error = finished["run_status"], finished["dump_complete"], finished["error"]
    require(status in ("completed", "failed") and type(claim) is bool and finished["qualified"] is False,
            "Finished line status, completeness or qualification is malformed")
    require(finished["planned_records"] == len(planned), "Finished line planned record count differs from the authenticated plan")
    require(finished["written_records"] == len(records), "Finished line record count differs from the progress records")
    require(finished["payload_bytes"] == payload_total, "Finished line payload bytes differ from the progress records")
    missing = finished["missing_records"]
    require(isinstance(missing, list) and missing == sorted(planned - set(records)),
            "Finished line missing records differ from the planned records not written")
    require(finished["incomplete_record"] == (started.get("key") if started else None),
            "Finished line incomplete record differs from the unfinished record start")
    require(error is None or isinstance(error, str) and error, "Finished line error must be null or a message")
    require((status == "failed") == (error is not None), "A completed run names no error and a failed run names one")
    complete = status == "completed" and not missing and started is None
    require(claim == complete, "Finished line dump_complete contradicts its own status, missing and incomplete records")
    consumed, digest = finished["consumed_rows"], finished["consumed_prefix_sha256"]
    if history is None:
        require(consumed == 0 and digest is None, "A dump without conditioning rows cannot claim consumed rows")
    else:
        require(natural(consumed) and len(history.rows) <= consumed, "Finished line consumed rows precede the dump's conditioning rows")
        require(len(history.rows) == consumed or not complete, "A complete dump must carry every consumed conditioning row")
        if len(history.rows) == consumed:
            require(digest == history.digests[consumed], "Finished line consumed-prefix SHA-256 differs from the conditioning rows")
        else:
            require(isinstance(digest, str) and DIGEST.fullmatch(digest), "Finished line consumed-prefix SHA-256 is malformed")
    if "byte_limit" in plan:
        require(payload_total <= plan["byte_limit"], "Payload bytes exceed the plan byte limit")
    if complete and "payload_bytes" in forecast:
        require(payload_total == forecast["payload_bytes"], "A complete dump must write exactly its forecast payload bytes")
    if "readbacks" in forecast:
        for field in ("readbacks", "readback_bytes"):
            require(natural(finished.get(field)) and finished[field] <= forecast[field]
                    and (finished[field] == forecast[field] or not complete),
                    f"Finished line {field} contradicts the plan forecast")
    else:
        require("readbacks" not in finished and "readback_bytes" not in finished, "Readback counts without a readback forecast")
    return complete


def load_dump(directory, allow_incomplete=False):
    """Authenticate a dump directory. Returns a dict; raises Refusal on any integrity failure."""
    root = Path(directory)
    require(root.is_dir() and not root.is_symlink(), f"Dump directory required: {root}")
    progress = root / "progress.jsonl"
    regular_file(progress, PROGRESS_LIMIT, "progress.jsonl")
    data = progress.read_bytes()
    require(data.endswith(b"\n"), "progress.jsonl ends with a partial line")
    lines = [strict_json(line.decode("utf-8"), f"progress line {i + 1}") for i, line in enumerate(data.split(b"\n")[:-1])]
    require(lines and lines[0].get("event") == "plan" and lines[0].get("schema") == PROGRESS_SCHEMA,
            f"progress.jsonl must begin with a {PROGRESS_SCHEMA} plan line")
    header = lines[0]
    commitments = header.get("commitments")
    require(header.get("qualified") is False, "A raw diagnostic dump cannot claim qualification")
    require(isinstance(commitments, dict) and isinstance(header.get("producer"), dict)
            and isinstance(commitments.get("model_revision"), str) and isinstance(commitments.get("input_manifest_sha256"), str),
            "Dump commitments must name the model revision and input manifest SHA-256")
    declared = header.get("conditioning")
    history = None
    if declared is not None:
        require(isinstance(declared, dict) and declared == {"schema": PREFIX_SCHEMA, "prompt_rows": declared.get("prompt_rows")}
                and natural(declared["prompt_rows"]) and declared["prompt_rows"] > 0, "Consumed-prefix declaration is malformed")
        inputs = commitments.get("input")
        prompt = inputs.get("prompt_tokens") if isinstance(inputs, dict) else None
        require(prompt is None or prompt == declared["prompt_rows"], "Consumed-prefix prompt rows differ from the input commitment")
        history = History(commitments["input_manifest_sha256"], declared["prompt_rows"])
    planned, forecast, plan = authenticate_plan(header, history.prompt_rows if history else None)
    model, records, files, started, finished, total = None, {}, set(), None, None, 0
    policy = {}
    for number, line in enumerate(lines[1:], start=2):
        where = f"progress line {number}"
        event = line.get("event")
        require(finished is None, f"{where}: content after the finished line")
        if event == "model_authenticated":
            require(model is None and not records and started is None, f"{where}: model commitment must precede records once")
            model = line.get("model")
            require(isinstance(model, dict) and model.get("revision") == commitments["model_revision"]
                    and isinstance(model.get("model_sha256"), str), f"{where}: model commitment differs")
        elif event == "boundary":
            require(started is None, f"{where}: boundary inside an unfinished record")
            require(line.get("completed_records") == len(records) and line.get("payload_bytes") == total,
                    f"{where}: boundary counts differ from the preceding records")
        elif event == "conditioning":
            require(started is None, f"{where}: conditioning rows inside an unfinished record")
            extend_history(history, line, where)
        elif event == "record_started":
            require(started is None, f"{where}: overlapping record writes")
            require(model is not None and line.get("key") in planned, f"{where}: record start is unplanned or precedes the model commitment")
            started = line
        elif event == "record":
            require(model is not None, f"{where}: record before model commitment")
            require(started is not None and started.get("sequence") == line.get("sequence") and started.get("file") == line.get("file")
                    and started.get("key") == line.get("key"), f"{where}: record without its matching start line")
            require(line.get("sequence") == len(records), f"{where}: record sequence is not contiguous")
            check_record(line, header, model, where)
            key = line["key"]
            require(key == record_key(line), f"{where}: record key differs from its identity fields")
            require(key not in records and line["file"] not in files, f"{where}: duplicate record identity or file")
            check_conditioning(line, history, where)
            size = regular_file(root / line["file"], PAYLOAD_LIMIT, "payload")
            require(size == line["payload_bytes"], f"{where}: payload file size differs")
            total += size
            require(total <= TOTAL_LIMIT and len(records) < RECORD_LIMIT, "Dump exceeds the comparator bounds")
            payload = (root / line["file"]).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == line["payload_sha256"], f"{where}: payload SHA-256 differs")
            seen = observe(line, payload)
            require(seen == line["observed"], f"{where}: declared observed counts differ from the payload")
            # A declared BF16 boundary is valid only if every finite value is exactly BF16; this is
            # independent of the other producer and of any caller tolerance.
            policy[key] = {"declared": line["bf16_rounding"], "finite_non_bf16_representable": seen["finite_non_bf16_representable"],
                           "valid": not (line["bf16_rounding"] == BF16_POLICY and seen["finite_non_bf16_representable"])}
            records[key] = line
            files.add(line["file"])
            started = None
        elif event == "finished":
            finished = line
        else:
            raise Refusal(f"{where}: unknown progress event {event!r}")
    status = finished.get("run_status") if finished else "unfinished"
    complete = bool(finished) and check_terminal(finished, planned, forecast, plan, records, started, total, history)
    orphans = sorted(p.name for p in root.iterdir() if p.name != "progress.jsonl" and p.name not in files)
    require(not (complete and orphans), f"Completed dump contains unreferenced files: {orphans[:5]}")
    require(allow_incomplete or complete,
            f"Dump is {status} and incomplete; pass --allow-incomplete to compare its completed records as unqualified evidence")
    return {"root": root, "header": header, "model": model or {}, "records": records, "finished": finished,
            "run_status": status, "dump_complete": complete, "planned_records": len(planned),
            "incomplete_record": started.get("key") if started else (finished or {}).get("incomplete_record"),
            "orphans": orphans, "progress_sha256": hashlib.sha256(data).hexdigest(), "history": history,
            "policy": policy, "policy_violations": sorted(k for k, v in policy.items() if not v["valid"])}


def values(dump, record):
    """Re-read and re-authenticate one payload, returning Python floats."""
    data = (dump["root"] / record["file"]).read_bytes()
    require(len(data) == record["payload_bytes"] and hashlib.sha256(data).hexdigest() == record["payload_sha256"],
            f"Payload changed after authentication: {record['file']}")
    code, _ = ITEM[record["storage_dtype"]]
    result = array(code)
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def compatible(candidate, reference):
    c, r = candidate["header"]["commitments"], reference["header"]["commitments"]
    for field in ("model_revision", "input_manifest_sha256"):
        require(c.get(field) == r.get(field), f"Incompatible {field}: candidate {c.get(field)} reference {r.get(field)}")
    for field in ("model_sha256", "config_sha256"):
        require(candidate["model"].get(field) == reference["model"].get(field) and candidate["model"].get(field),
                f"Incompatible authenticated {field}")


def identity_difference(c, r):
    for field in IDENTITY:
        if c.get(field) != r.get(field):
            return f"{field} differs"
    a, b = c["coordinates"][0], r["coordinates"][0]
    for field in SEMANTIC:
        if (field in a or field in b) and a.get(field) != b.get(field):
            return f"coordinate 0 {field} differs"
    if c["conditioning"] != r["conditioning"]:
        return "consumed prefix differs"
    return None


def representable_bf16(values_):
    """Monotonic integer BF16 codes, or None if any finite value is not exactly BF16."""
    f32 = array("f", values_)
    if any(a != b for a, b in zip(f32, values_)):
        return None  # A float64 value outside FP32 cannot be a BF16 boundary value.
    words = array(WORD)
    words.frombytes(f32.tobytes())
    if any(w & 0xFFFF for w in words):
        return None
    return [-((w >> 16) & 0x7FFF) if w & 0x80000000 else w >> 16 for w in words]


def argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def rank(row, index):
    target = row[index]
    return sum(1 for i, v in enumerate(row) if v > target or (v == target and i < index))


def top_tokens(c, r, max_abs):
    ct, rt = argmax(c), argmax(r)

    def margin(row, top):
        second = max(v for i, v in enumerate(row) if i != top)
        return row[top] - second, sum(1 for v in row if v == row[top])

    cm, cties = margin(c, ct)
    rm, rties = margin(r, rt)
    top5 = lambda row: [i for _, i in sorted(((-v, i) for i, v in enumerate(row)))[:5]]
    return {"candidate_top": ct, "reference_top": rt, "argmax_agree": ct == rt, "candidate_margin": cm,
            "reference_margin": rm, "candidate_max_ties": cties, "reference_max_ties": rties,
            "candidate_rank_of_reference_top": rank(c, rt), "reference_rank_of_candidate_top": rank(r, ct),
            "candidate_logit_at_reference_top": c[rt], "top5_overlap": len(set(top5(c)) & set(top5(r))),
            "reference_margin_exceeds_twice_max_abs_error": max_abs is not None and rm > 2 * max_abs,
            "tie_policy": "first vocabulary index"}


def metrics(c_record, r_record, c, r, candidate_valid=True):
    n = len(c)
    result = {"elements": n, "candidate_nonfinite": sum(1 for v in c if not math.isfinite(v)),
              "reference_nonfinite": sum(1 for v in r if not math.isfinite(v)),
              "exact_equal": sum(1 for a, b in zip(c, r) if a == b)}
    result["finite"] = result["candidate_nonfinite"] == 0 and result["reference_nonfinite"] == 0
    if not result["finite"]:
        return result
    diffs = [abs(a - b) for a, b in zip(c, r)]
    worst = max(range(n), key=diffs.__getitem__)
    rel = [d / abs(b) for d, b in zip(diffs, r) if b != 0]
    nc, nr = math.sqrt(math.fsum(a * a for a in c)), math.sqrt(math.fsum(b * b for b in r))
    result.update({
        "max_abs": diffs[worst], "max_abs_index": worst, "max_rel": max(rel) if rel else 0.0,
        "zero_reference_elements": n - len(rel),
        "zero_reference_nonzero_candidate": sum(1 for a, b in zip(c, r) if b == 0 and a != 0),
        "rmse": math.sqrt(math.fsum(d * d for d in diffs) / n), "reference_rms": nr / math.sqrt(n),
        "cosine": math.fsum(a * b for a, b in zip(c, r)) / (nc * nr) if nc > 0 and nr > 0 else None,
    })
    if c_record["bf16_rounding"] == BF16_POLICY and r_record["bf16_rounding"] == BF16_POLICY:
        oc, orr = representable_bf16(c), representable_bf16(r)
        if not candidate_valid or oc is None or orr is None:
            result["bf16"] = {"applicable": False, "reason": "a declared BF16 boundary value is not BF16 representable"}
        else:
            ulps = [abs(a - b) for a, b in zip(oc, orr)]
            result["bf16"] = {"applicable": True, "max_ulp": max(ulps), "exact": ulps.count(0), "one_ulp": ulps.count(1),
                              "two_ulp": ulps.count(2), "over_two_ulp": sum(1 for u in ulps if u > 2)}
    else:
        result["bf16"] = {"applicable": False, "reason": "at least one record is not a declared BF16 boundary"}
    if c_record["stage"] == "text.logits" and c_record["payload_shape"] == [1, VOCABULARY] and c_record["complete_tensor"]:
        result["top_tokens"] = top_tokens(c, r, result["max_abs"])
    return result


def verdict(c_record, r_record, m, tolerances):
    if not m["finite"]:
        return ("outside_tolerance" if m["candidate_nonfinite"] else "incomparable"), ["nonfinite values"]
    criteria = dict(tolerances.get("*", {}))
    criteria.update(tolerances.get(c_record["stage"], {}))
    # BF16 ULP applies only where both producers declare a BF16 graph boundary (FP32 state is exempt);
    # argmax applies only to complete vocabulary rows.
    if not (c_record["bf16_rounding"] == BF16_POLICY and r_record["bf16_rounding"] == BF16_POLICY):
        criteria.pop("max_bf16_ulp", None)
    if "top_tokens" not in m:
        criteria.pop("require_argmax_match", None)
    if not criteria:
        return "no_tolerance", []
    failed = []
    if "max_abs" in criteria and m["max_abs"] > criteria["max_abs"]:
        failed.append("max_abs")
    if "max_rel" in criteria and (m["max_rel"] > criteria["max_rel"] or m["zero_reference_nonzero_candidate"]):
        failed.append("max_rel")
    if "max_rmse" in criteria and m["rmse"] > criteria["max_rmse"]:
        failed.append("max_rmse")
    if "min_cosine" in criteria and not (m["cosine"] is not None and m["cosine"] >= criteria["min_cosine"]
                                         or m["cosine"] is None and m["exact_equal"] == m["elements"]):
        failed.append("min_cosine")
    if "max_bf16_ulp" in criteria and not (m["bf16"]["applicable"] and m["bf16"]["max_ulp"] <= criteria["max_bf16_ulp"]):
        failed.append("max_bf16_ulp")
    if criteria.get("require_argmax_match") and "top_tokens" in m and not m["top_tokens"]["argmax_agree"]:
        failed.append("require_argmax_match")
    return ("outside_tolerance" if failed else "within_tolerance"), failed


def conditioning_summary(candidate, reference):
    c, r = candidate["history"], reference["history"]
    if c is None or r is None:
        return None
    common = min(len(c.rows), len(r.rows))
    first = next((i for i in range(common) if c.rows[i] != r.rows[i]), None)
    same_prompt = c.prompt_rows == r.prompt_rows
    return {"candidate_prompt_rows": c.prompt_rows, "reference_prompt_rows": r.prompt_rows,
            "candidate_rows": len(c.rows), "reference_rows": len(r.rows), "first_divergent_row": first,
            "first_divergent_generated_index": first - c.prompt_rows if first is not None and same_prompt and first >= c.prompt_rows else None,
            "common_rows_identical": first is None and same_prompt}


def compare(candidate, reference, tolerances):
    compatible(candidate, reference)
    rows, counts = [], {}
    for key, c_record in candidate["records"].items():
        r_record = reference["records"].get(key)
        c_policy = candidate["policy"][key]
        row = {"key": key, "stage": c_record["stage"], "file": c_record["file"],
               "policy": {"candidate": c_policy, "reference": reference["policy"].get(key)}}
        reason = "reference record absent" if r_record is None else identity_difference(c_record, r_record)
        if not reason and not reference["policy"][key]["valid"]:
            reason = "reference declares a BF16 boundary holding finite non-BF16 values"
        if reason:
            row.update(status="incomparable", reason=reason)
        else:
            m = metrics(c_record, r_record, values(candidate, c_record), values(reference, r_record), c_policy["valid"])
            status, failed = verdict(c_record, r_record, m, tolerances)
            if c_policy["valid"]:
                row.update(status=status, failed_criteria=failed, metrics=m)
            else:
                # A candidate that breaks its own declared BF16 boundary fails whatever tolerances were chosen.
                row.update(status="invalid_candidate", tolerance_status=status, failed_criteria=failed, metrics=m,
                           reason=f"candidate declares a BF16 boundary holding {c_policy['finite_non_bf16_representable']} finite non-BF16 values")
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        rows.append(row)
    kind = str(reference["header"]["producer"].get("kind", ""))
    clean = bool(rows) and counts.get("within_tolerance", 0) == len(rows)
    complete = candidate["dump_complete"] and reference["dump_complete"]
    violations = candidate["policy_violations"] or reference["policy_violations"]
    overall = ("incomparable" if counts.get("incomparable") else
               "invalid_candidate" if counts.get("invalid_candidate") or candidate["policy_violations"] else
               "outside_tolerance" if counts.get("outside_tolerance") else
               "no_verdict" if not clean else "within_tolerance" if complete else "within_tolerance_incomplete_dump")
    return {
        "schema": COMPARISON_SCHEMA, "verdict": overall, "counts": counts, "tolerances": tolerances,
        "candidate": summary(candidate), "reference": summary(reference),
        "extra_reference_records": len(set(reference["records"]) - set(candidate["records"])),
        "conditioning": conditioning_summary(candidate, reference),
        "candidate_policy_violations": candidate["policy_violations"],
        "reference_policy_violations": reference["policy_violations"],
        "reference_kind": kind, "reference_independent": candidate["header"]["producer"] != reference["header"]["producer"],
        "cpu_oracle_agreement": clean and complete and not violations and kind.startswith("cpu_"),
        "amd_ocr_qualified": False, "native_numerically_qualified": False,
        "scope": ("Selected graph-boundary agreement with the supplied reference under caller tolerances only, for "
                  "records whose complete consumed prefixes are identical. A CPU oracle match is operator evidence; it "
                  "is not primary AMD OCR qualification, complete-page acceptance or throughput. Incomplete dumps and "
                  "dumps with declared-policy violations remain unqualified."),
        "records": rows,
    }


def summary(dump):
    history = dump["history"]
    return {"directory": str(dump["root"]), "progress_sha256": dump["progress_sha256"], "producer": dump["header"]["producer"],
            "plan_sha256": dump["header"]["plan_sha256"], "run_status": dump["run_status"], "dump_complete": dump["dump_complete"],
            "planned_records": dump["planned_records"], "records": len(dump["records"]),
            "incomplete_record": dump["incomplete_record"], "unreferenced_files": dump["orphans"],
            "conditioning_rows": len(history.rows) if history else 0}


def write_dump(directory, commitments, model, producer, records, plan=None, run_completed=True, *,
               conditioning=None, prompt_rows=None, planned_keys=None, error=None):
    """Write a dump in the native format, e.g. for a retained CPU reference. Fails if the directory exists.

    records: dicts with the record identity fields (stage, phase, layer, vision_block, decode_step, cache_length,
    logical_shape, selected_rows, payload_shape, complete_tensor, coordinates, bf16_rounding, decode,
    produces_generated_index, cache) plus "values" and optional "storage_dtype" (float32 default).
    conditioning/prompt_rows: the producer's actual consumed request rows [token_id, t, h, w] in absolute order
    and its prompt length; required for causal text records, whose prefix commitments are derived from them.
    plan: optional descriptive fields; planned_keys: the producer's planned identities (default: those written).
    """
    root = Path(directory)
    history = None
    if conditioning is not None or prompt_rows is not None:
        if not (natural(prompt_rows) and prompt_rows > 0 and isinstance(conditioning, list)):
            raise ValueError("conditioning rows and a positive prompt_rows must be supplied together")
        history = History(commitments["input_manifest_sha256"], prompt_rows)
        for row in conditioning:
            history.append(row)
    full = dict(commitments)
    full.update({k: model[k] for k in ("model_sha256", "config_sha256", "model_bytes") if k in model})
    prepared = []
    for sequence, spec in enumerate(records):
        spec = dict(spec)
        dtype = spec.pop("storage_dtype", "float32")
        data = array(ITEM[dtype][0], spec.pop("values"))
        if sys.byteorder != "little":
            data.byteswap()
        payload = data.tobytes()
        given = spec.pop("key", None)
        record = {"event": "record", "schema": RECORD_SCHEMA, "selected_axis": 0, "layer": None, "vision_block": None,
                  "decode_step": None, "cache_length": None, "decode": None, "produces_generated_index": None,
                  "cache": None, "bf16_rounding": BF16_POLICY, "conditioning": None}
        record.update(spec)
        key = record_key(record)
        if given is not None and given != key:
            raise ValueError(f"record key {given!r} differs from its identity {key!r}")
        if record["stage"] in CONDITIONED:
            if history is None:
                raise ValueError("causal text records require the producer's actual consumed conditioning rows and prompt_rows")
            n = record["cache_length"] if record["stage"] in STATE_STAGES else record["coordinates"][0]["absolute_row"] + 1
            if not (natural(n) and 1 <= n <= len(history.rows)):
                raise ValueError(f"{key}: consumed prefix of {n} rows is not covered by the conditioning rows")
            derived = {"prefix_rows": n, "prefix_sha256": history.digests[n]}
            if record["conditioning"] not in (None, derived):
                raise ValueError(f"{key}: supplied conditioning differs from the producer's consumed rows")
            record["conditioning"] = derived
        elif record["conditioning"] is not None:
            raise ValueError(f"{key}: only causal text records carry a consumed-prefix commitment")
        name = f"{sequence:05d}.{record['stage']}.f{8 * ITEM[dtype][1]}"
        record.update(sequence=sequence, file=name, key=key, storage_dtype=dtype, byte_order="little",
                      payload_bytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest(),
                      observed=observe({"storage_dtype": dtype}, payload), qualified=False)
        prepared.append((record, payload))
    written = [record["key"] for record, _ in prepared]
    if len(set(written)) != len(written):
        raise ValueError("records repeat an identity")
    planned = sorted(set(written) | set(planned_keys or ()))
    description = dict(plan or {})
    if {"schema", "forecast"} & set(description):
        raise ValueError("write_dump sets the reference plan schema and forecast itself")
    plan = dict(description, schema=REFERENCE_PLAN_SCHEMA, forecast={"records": len(planned), "keys": planned})
    plan_sha = canonical_sha256(plan)
    if full.setdefault("plan_sha256", plan_sha) != plan_sha:
        raise ValueError("commitments name a different plan_sha256")
    for record, _ in prepared:
        record.update(commitments=full, producer=producer)
    os.mkdir(root, 0o700)

    def create(name, data):
        fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    lines = [{"event": "plan", "schema": PROGRESS_SCHEMA, "plan": plan, "plan_sha256": plan_sha, "commitments": full,
              "producer": producer, "qualified": False,
              "conditioning": {"schema": PREFIX_SCHEMA, "prompt_rows": prompt_rows} if history else None},
             {"event": "model_authenticated", "model": dict(model, revision=commitments["model_revision"])}]
    if history and history.rows:
        lines.append({"event": "conditioning", "first_row": 0, "rows": history.rows, "consumed_rows": len(history.rows),
                      "prefix_sha256": history.digests[-1]})
    for record, payload in prepared:
        lines.append({"event": "record_started", "sequence": record["sequence"], "file": record["file"], "key": record["key"]})
        create(record["file"], payload)
        lines.append(record)
    missing = sorted(set(planned) - set(written))
    status = "completed" if run_completed and not error else "failed"
    lines.append({"event": "finished", "run_status": status, "dump_complete": status == "completed" and not missing,
                  "planned_records": len(planned), "written_records": len(prepared), "missing_records": missing,
                  "incomplete_record": None, "payload_bytes": sum(len(p) for _, p in prepared),
                  "error": None if status == "completed" else error or "reference producer reported failure",
                  "consumed_rows": len(history.rows) if history else 0,
                  "consumed_prefix_sha256": history.digests[-1] if history else None, "qualified": False})
    create("progress.jsonl", "".join(json.dumps(line, allow_nan=False) + "\n" for line in lines).encode())
    return root


def tolerance_arguments(args):
    tolerances = {}
    # Zero is a meaningful tolerance; only an absent argument means "not supplied".
    common = {name: getattr(args, name) for name in CRITERIA if getattr(args, name) is not None and getattr(args, name) is not False}
    if common:
        tolerances["*"] = common
    for item in args.stage_tolerance or []:
        stage, _, rest = item.partition(":")
        require(stage and rest, f"Stage tolerance must be STAGE:criterion=value[,criterion=value]: {item}")
        entry = tolerances.setdefault(stage, {})
        for part in rest.split(","):
            name, _, value = part.partition("=")
            require(name in CRITERIA, f"Unknown tolerance criterion {name!r}")
            if name == "require_argmax_match":
                require(value in ("true", "false"), "require_argmax_match accepts true or false")
                entry[name] = value == "true"
            else:
                number = float(value)
                require(math.isfinite(number) and number >= 0 or name == "min_cosine" and -1 <= number <= 1,
                        f"Tolerance {name} must be finite and nonnegative")
                entry[name] = int(number) if name == "max_bf16_ulp" else number
    return tolerances


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", type=Path, help="native diagnostic directory")
    parser.add_argument("reference", type=Path, help="retained reference dump directory")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="compare completed records of failed, unfinished or incomplete dumps as unqualified evidence")
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--max-rel", type=float)
    parser.add_argument("--max-rmse", type=float)
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--max-bf16-ulp", type=int)
    parser.add_argument("--require-argmax-match", action="store_true")
    parser.add_argument("--stage-tolerance", action="append", metavar="STAGE:criterion=value,...")
    parser.add_argument("--output", type=Path, help="fresh report file; never overwritten")
    args = parser.parse_args(argv)
    try:
        for name in ("max_abs", "max_rel", "max_rmse"):
            value = getattr(args, name)
            require(value is None or math.isfinite(value) and value >= 0, f"--{name.replace('_', '-')} must be finite and nonnegative")
        require(args.min_cosine is None or -1 <= args.min_cosine <= 1, "--min-cosine must be within [-1, 1]")
        require(args.max_bf16_ulp is None or args.max_bf16_ulp >= 0, "--max-bf16-ulp must be nonnegative")
        require(args.output is None or not os.path.lexists(args.output), "--output must name a fresh file; reports are never overwritten")
        tolerances = tolerance_arguments(args)
        report = compare(load_dump(args.candidate, args.allow_incomplete), load_dump(args.reference, args.allow_incomplete), tolerances)
    except Refusal as error:
        print(json.dumps({"schema": COMPARISON_SCHEMA, "verdict": "refused", "error": str(error),
                          "amd_ocr_qualified": False, "native_numerically_qualified": False}))
        return 2
    text = json.dumps(report, allow_nan=False, indent=1) + "\n"
    if args.output:
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    return {"within_tolerance": 0, "no_verdict": 3, "within_tolerance_incomplete_dump": 3}.get(report["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
