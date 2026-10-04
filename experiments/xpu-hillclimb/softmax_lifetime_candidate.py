"""Prospective text eager softmax lifetime experiment; inactive unless scoped."""
import ast
import __future__
from contextlib import contextmanager
import hashlib
from pathlib import Path

IDENTITY = "text-eager-softmax-lifetime-v1"
GRAPH_SHA256 = "d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939"


def build_candidate(graph):
    """Compile the pinned stock function with only its compound cast split."""
    source = Path(graph.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != GRAPH_SHA256:
        raise RuntimeError("Installed graph source differs from pinned graph")
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "eager_attention_forward")
    target = ast.parse("attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)").body[0]
    matches = [i for i, n in enumerate(function.body) if ast.dump(n) == ast.dump(target)]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one pinned compound softmax cast")
    replacement = ast.parse("""softmax_fp32 = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
del attn_weights
attn_weights = softmax_fp32.to(query.dtype)
del softmax_fp32
""").body
    i = matches[0]
    function.body[i:i + 1] = replacement
    function.decorator_list = []
    namespace = dict(vars(graph))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(graph.__file__), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["eager_attention_forward"]


@contextmanager
def scoped_candidate(graph):
    """Intercept exact text BF16 evaluation only; restore on every exit."""
    original = graph.eager_attention_forward
    candidate = build_candidate(graph)
    counts = {"intercepted": 0, "forwarded": 0}

    def dispatch(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if (type(module) is graph.Qwen3_5Attention and not module.training
                and not graph.torch.is_grad_enabled()
                and query.dtype == key.dtype == value.dtype == graph.torch.bfloat16):
            counts["intercepted"] += 1
            return candidate(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        counts["forwarded"] += 1
        return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    graph.eager_attention_forward = dispatch
    try:
        yield counts
    finally:
        graph.eager_attention_forward = original
