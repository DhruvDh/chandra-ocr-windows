"""Translate unmodified runtime HLSL compute shaders into C++ structs for ChandraNative/runtime/hlsl_cpu.h.

Test-only and standard library only. The translation is purely syntactic: resources become RAW buffer views,
the b0 cbuffer becomes member fields loaded with HLSL packing, groupshared arrays become race-tracked
shared arrays, every GroupMemoryBarrierWithGroupSync() becomes a numbered barrier call and FP literals gain
an f suffix so arithmetic stays FP32. Anything outside the small subset used by ChandraNative/shaders/runtime
is refused rather than guessed.
"""
import hashlib
import re
from pathlib import Path

SEMANTICS = {"SV_GroupID": "group", "SV_GroupIndex": "index", "SV_DispatchThreadID": "dispatch", "SV_GroupThreadID": "groupThread"}
FLOAT_LITERAL = re.compile(r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])")
UNSUPPORTED = (r"AllMemoryBarrier\w*", r"DeviceMemoryBarrier\w*", r"Interlocked\w*", r"\w*StructuredBuffer", r"\w*Texture\w*",
               r"SamplerState", r"Wave\w+", r"half\d*", r"double\d*", r"min16\w*", r"goto", r"static", r"groupshared\s+\w+\s+\w+\s*=")


def inline(path, seen=None):
    path = Path(path)
    seen = set() if seen is None else seen
    if path.resolve() in seen:
        raise ValueError(f"recursive include {path}")
    seen.add(path.resolve())
    out = []
    for line in path.read_text().splitlines():
        m = re.match(r'\s*#include\s+"([^"]+)"\s*$', line)
        out.append(inline(path.parent / m.group(1), seen) if m else line)
    return "\n".join(out)


def _fields(body):
    fields = []
    for decl in body.split(";"):
        decl = decl.strip()
        if not decl:
            continue
        m = re.match(r"(uint|float|uint3|uint4)\s+(.+)$", decl, re.S)
        if not m:
            raise ValueError(f"unsupported cbuffer field: {decl!r}")
        for name in m.group(2).split(","):
            fields.append((m.group(1), name.strip()))
    offset, out = 0, []
    for kind, name in fields:
        size = {"uint": 4, "float": 4, "uint3": 12, "uint4": 16}[kind]
        if offset % 16 + size > 16:
            offset = (offset + 15) // 16 * 16
        out.append((kind, name, offset, size))
        offset += size
    if offset > 256:
        raise ValueError("cbuffer exceeds the Device's 256-byte constants")
    return out


def translate(text, struct, shader_name):
    """Return C++ source defining struct `struct` (namespace hlsl_cpu) and its Entry."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    for pattern in UNSUPPORTED:
        if re.search(r"\b" + pattern + r"\b", text):
            raise ValueError(f"{shader_name}: unsupported HLSL construct {pattern!r}")
    define_lines = re.findall(r"^\s*#define\s+(\w+)\s+(.+?)\s*$", text, re.M)
    if len(define_lines) != len(re.findall(r"#define", text)):
        raise ValueError(f"{shader_name}: unsupported preprocessor definition")
    text = re.sub(r"^\s*#define[^\n]*$", "", text, flags=re.M)
    values = {name: value for name, value in define_lines}

    def integer(token):
        token = values.get(token, token)
        if not re.fullmatch(r"\d+", token):
            raise ValueError(f"{shader_name}: numthreads needs integer literals or integer defines")
        return token
    threads = [tuple(integer(v) for v in t) for t in re.findall(r"\[numthreads\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)\]", text)]
    if len(threads) != 1:
        raise ValueError(f"{shader_name}: exactly one numthreads attribute required")
    text = re.sub(r"\[(numthreads|unroll|loop)(\([^)]*\))?\]", "", text)
    srv, uav, members = {}, {}, []

    def resource(m):
        rw, name, kind, slot = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if (kind == "u") != bool(rw):
            raise ValueError(f"{shader_name}: {name} register class differs from its view type")
        table = uav if rw else srv
        if slot in table:
            raise ValueError(f"{shader_name}: duplicate register {kind}{slot}")
        table[slot] = name
        members.append(f"{'RWByteAddressBuffer' if rw else 'ByteAddressBuffer'} {name};")
        return ""
    text = re.sub(r"(RW)?ByteAddressBuffer\s+(\w+)\s*:\s*register\(\s*([tu])(\d+)\s*\)\s*;", resource, text)
    if "ByteAddressBuffer" in re.sub(r"\(\s*(RW)?ByteAddressBuffer\s+\w+|,\s*(RW)?ByteAddressBuffer\s+\w+", "", text):
        raise ValueError(f"{shader_name}: unregistered resource declaration")
    cbuffers = re.findall(r"cbuffer\s+\w+\s*:\s*register\(\s*b(\d+)\s*\)\s*\{([^}]*)\}\s*;?", text)
    if len(cbuffers) > 1 or (cbuffers and cbuffers[0][0] != "0"):
        raise ValueError(f"{shader_name}: only one b0 cbuffer is supported")
    fields = _fields(cbuffers[0][1]) if cbuffers else []
    text = re.sub(r"cbuffer\s+\w+\s*:\s*register\(\s*b0\s*\)\s*\{[^}]*\}\s*;?", "", text)
    shared = []

    def groupshared(m):
        kind, out = m.group(1), []
        for decl in m.group(2).split(","):
            dm = re.match(r"\s*(\w+)\s*\[([^\]]+)\]\s*(?:\[([^\]]+)\])?\s*$", decl)
            if not dm:
                raise ValueError(f"{shader_name}: unsupported groupshared declaration {decl!r}")
            name, n, k = dm.groups()
            shared.append(name)
            out.append(f"Shared2<{kind},{n},{k}> {name};" if k else f"Shared<{kind},{n}> {name};")
        members.extend(out)
        return ""
    text = re.sub(r"groupshared\s+(float|uint)\s+([^;]+);", groupshared, text)
    text = re.sub(r"\bprecise\b", "", text)
    counter = iter(range(1, 10000))
    text = re.sub(r"GroupMemoryBarrierWithGroupSync\s*\(\s*\)", lambda m: f"hlsl_barrier({next(counter)})", text)
    barriers = "hlsl_barrier(" in text
    mains = re.findall(r"void\s+main\s*\(([^)]*)\)", text)
    if len(mains) != 1:
        raise ValueError(f"{shader_name}: one main entry required")
    params, args = [], []
    for raw in mains[0].split(","):
        pm = re.match(r"\s*(uint3|uint)\s+(\w+)\s*:\s*(\w+)\s*$", raw)
        if not pm or pm.group(3) not in SEMANTICS:
            raise ValueError(f"{shader_name}: unsupported entry parameter {raw!r}")
        kind, name, semantic = pm.groups()
        params.append(f"{kind} {name}")
        value = f"hlsl_id.{SEMANTICS[semantic]}"
        args.append(value + (".x" if kind == "uint" and semantic != "SV_GroupIndex" else ""))
    text = re.sub(r"void\s+main\s*\([^)]*\)", "void main_(" + ", ".join(params) + ")", text)
    text = FLOAT_LITERAL.sub(lambda m: m.group(1) + "f", text)
    defines = [name for name, _ in define_lines]
    constants = "".join(f"std::memcpy(&{name},reinterpret_cast<const char*>(hlsl_constants)+{offset},{size});" for _, name, offset, size in fields)
    binds = "".join(f'{name}.m=slot(hlsl_binding.srv,{s});{name}.uav=false;{name}.slot="t{s}";' for s, name in sorted(srv.items()))
    binds += "".join(f'{name}.m=slot(hlsl_binding.uav,{s});{name}.uav=true;{name}.slot="u{s}";' for s, name in sorted(uav.items()))
    resets = "".join(f'{name}.reset("{name}");' for name in shared)
    field_decls = "".join(f"{kind} {name}{{}};" for kind, name, _, _ in fields)
    x, y, z = threads[0]
    source = (f"struct {struct} : Shader {{\n"
              f"static constexpr const char* name=\"{shader_name}\";\n"
              f"static constexpr uint3 threads{{{x},{y},{z}}};\n"
              f"static constexpr bool barriers={'true' if barriers else 'false'};\n"
              + "".join(f"#define {n} {v}\n" for n, v in define_lines) +
              f"{' '.join(members)}\n{field_decls}\n"
              f"static Memory* slot(const std::vector<Memory*>& v,size_t i) {{ return i<v.size()?v[i]:nullptr; }}\n"
              f"void bind(const Binding& hlsl_binding) {{ {binds} }}\n"
              f"void constants(const uint32_t* hlsl_constants) {{ (void)hlsl_constants;{constants} }}\n"
              f"void reset() {{ {resets} }}\n"
              f"void invoke(const ThreadId& hlsl_id) {{ main_({', '.join(args)}); }}\n"
              f"#line 1 \"{shader_name}\"\n{text}\n}};\n")
    source += "".join(f"#undef {d}\n" for d in defines)
    entry = (f"Entry{{{struct}::name,&run<{struct}>,{struct}::barriers,{struct}::threads,"
             f"{max(srv) + 1 if srv else 0}u,{max(uav) + 1 if uav else 0}u}}")
    return source, entry


def generate(shaders, output, extra=()):
    """Translate runtime shaders (paths below a shader root, named runtime/<file>) plus optional (name, path)
    pairs, and write one header with an `entries()` registry. Returns {name: sha256 of the original bytes}."""
    parts, entries, digests = ['#pragma once\n#include "hlsl_cpu.h"\nnamespace hlsl_cpu {\n'], [], {}
    for index, (name, path) in enumerate(list(shaders) + list(extra)):
        digests[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        source, entry = translate(inline(path), f"S{index}", name)
        parts.append(source)
        entries.append(entry)
    parts.append("inline const std::vector<Entry>& entries() { static const std::vector<Entry> e{" + ",".join(entries) + "};return e; }\n}\n")
    Path(output).write_text("\n".join(parts))
    return digests
