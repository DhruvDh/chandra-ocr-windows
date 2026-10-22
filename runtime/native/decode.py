"""Incremental detokenization that never splits a character and must agree with one full decode."""


class IncrementalDecoder:
    """Emit text only at complete-character boundaries, as the pinned tokenizer decodes the whole sequence.

    decode(ids) must be the pinned tokenizer's decode(ids, skip_special_tokens=True,
    clean_up_tokenization_spaces=False). A window that ends inside a multi-byte UTF-8 sequence decodes
    to a trailing U+FFFD, so it is held until a later token completes it. finish() compares the
    streamed text with a single full decode; any disagreement is a failure, never silently patched.
    """

    def __init__(self, decode):
        self.decode = decode
        self.ids = []
        self.prefix = 0
        self.read = 0
        self.parts = []

    def push(self, token_id):
        self.ids.append(token_id)
        before = self.decode(self.ids[self.prefix:self.read]) if self.read > self.prefix else ""
        after = self.decode(self.ids[self.prefix:])
        if len(after) <= len(before) or after.endswith("�") or not after.startswith(before):
            return ""
        delta = after[len(before):]
        self.prefix, self.read = self.read, len(self.ids)
        self.parts.append(delta)
        return delta

    def finish(self):
        text = self.decode(self.ids)
        streamed = "".join(self.parts)
        if not text.startswith(streamed):
            raise RuntimeError("Incremental detokenization disagrees with the complete decode")
        rest = text[len(streamed):]
        if rest:
            self.parts.append(rest)
        return rest

    @property
    def text(self):
        return "".join(self.parts)
