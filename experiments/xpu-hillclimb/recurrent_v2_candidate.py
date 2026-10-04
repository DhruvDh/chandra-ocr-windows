"""Experimental source-identical recurrence with contraction disabled.

Supported pinned compiler options change arithmetic policy. Numerical equivalence is unproven.
Use exclusively in an owned experimental process with a fresh code cache.
"""
from contextlib import contextmanager
import hashlib
import inspect
import os

from recurrent_candidate import compiled_recurrent as v1_scope


PINNED_SOURCES = {
    "torch._inductor.codegen.triton": "f533949df423872849aac71a9375cf2a9d56d246340cbec7be749bc8dd5045c4",
    "torch._inductor.config": "76e5829739918f8ad7a8c0e1ac491d39fd0ab14faf642afe534a8ee9d1a03484",
    "triton.backends.intel.compiler": "6e94473952ca0fa39f951003a89c319645a7b13c51e38a004598d4995ca543aa",
    "triton.knobs": "338c3fec0d0503982d201df6f75e91eb874b0a7a5e370f9deedcde2fd1b703bf",
}


@contextmanager
def compiled_recurrent(*, expected_graph_sha256, fixture=None, compile_function=None):
    import importlib
    import torch
    from torch._inductor.codegen.triton import TritonKernel
    from triton import knobs
    from triton.backends.intel.compiler import XPUBackend

    if compile_function is not None:
        raise RuntimeError("V2 requires its pinned compiler controls")
    hashes = {}
    for name, expected in PINNED_SOURCES.items():
        module = importlib.import_module(name)
        with open(inspect.getfile(module), "rb") as stream:
            hashes[name] = hashlib.sha256(stream.read()).hexdigest()
        if hashes[name] != expected:
            raise RuntimeError(f"Unapproved compiler source: {name}")
    original_descriptor = TritonKernel.__dict__["triton_meta_common"]
    original_parse_options = XPUBackend.parse_options
    env_names = ("TRITON_INTEL_FAST_MATH", "TORCHINDUCTOR_USE_FAST_MATH",
                 "TRITON_DEFAULT_FP_FUSION")
    old_environment = {name: os.environ.get(name) for name in env_names}
    controls = {"enable_fp_fusion": False, "use_fast_math": False,
                "emulate_precision_casts": True, "metadata_emissions": 0,
                "verified_backend_options": 0,
                "compiler_source_sha256": hashes, "restored": False}

    def metadata(cls):
        result = dict(original_descriptor.__get__(None, cls)())
        if result["enable_fp_fusion"]:
            raise RuntimeError("Supported precision option did not disable contraction")
        controls["metadata_emissions"] += 1
        return result

    def compiler(function, **kwargs):
        return torch.compile(function, **kwargs,
                             options={"use_fast_math": False,
                                      "emulate_precision_casts": True})

    def parse_options(backend, options):
        result = original_parse_options(backend, options)
        if result.enable_fp_fusion or knobs.intel.fast_math:
            raise RuntimeError("Intel compiler received contraction/fast-math enabled")
        controls["verified_backend_options"] += 1
        return result

    try:
        for name in env_names:
            os.environ[name] = "0"
        if knobs.intel.fast_math:
            raise RuntimeError("Pinned Intel fast-math knob did not resolve False")
        TritonKernel.triton_meta_common = classmethod(metadata)
        XPUBackend.parse_options = parse_options
        with v1_scope(expected_graph_sha256=expected_graph_sha256,
                      fixture=fixture, compile_function=compiler) as receipt:
            receipt["candidate"] = "source-identical-compiled-recurrent-decode-v2-no-contract"
            receipt["compiler_controls"] = controls
            yield receipt
            if receipt["calls"] and (not controls["metadata_emissions"] or
                                     not controls["verified_backend_options"]):
                raise RuntimeError("V2 compilation controls were not observed")
    finally:
        TritonKernel.triton_meta_common = original_descriptor
        XPUBackend.parse_options = original_parse_options
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        controls["restored"] = True
