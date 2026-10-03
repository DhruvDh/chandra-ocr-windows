import base64
import json
import re

MAX_BYTES = 128 * 1024 * 1024

def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    allowed = {"model", "messages", "stream", "temperature", "top_p", "max_tokens", "stream_options"}
    if set(payload) - allowed:
        raise ValueError("Unsupported request fields: " + ", ".join(sorted(set(payload) - allowed)))
    options = payload.get("stream_options", {})
    if not isinstance(options, dict) or set(options) - {"include_usage"} or type(options.get("include_usage", False)) is not bool:
        raise ValueError("Only stream_options.include_usage boolean is supported.")
    if payload.get("model") != "chandra":
        raise ValueError("Use model alias chandra.")
    if type(payload.get("stream", False)) is not bool:
        raise ValueError("stream must be boolean.")
    if type(payload.get("temperature", 0)) not in {int, float} or payload.get("temperature", 0) != 0:
        raise ValueError("Only deterministic temperature=0 is supported.")
    top_p = payload.get("top_p", 1)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    tokens = payload.get("max_tokens", 12384)
    if type(tokens) is not int or not 1 <= tokens <= 12384:
        raise ValueError("max_tokens must be an integer between 1 and 12384.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Supply exactly one user message containing text and one image data URL.")
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != "user":
        raise ValueError("Only a user message with role and content is supported.")
    content = message["content"]
    if not isinstance(content, list) or len(content) != 2:
        raise ValueError("Supply one text part and one image_url part.")
    kinds = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Invalid message part.")
        kinds.append(part.get("type"))
        if part.get("type") == "text":
            if set(part) != {"type", "text"} or not isinstance(part["text"], str) or not part["text"] or len(part["text"]) > 16384:
                raise ValueError("Text must be nonempty and at most 16384 characters.")
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            if set(part) != {"type", "image_url"} or not isinstance(image, dict) or set(image) != {"url"}:
                raise ValueError("image_url must contain only url.")
            url = image["url"]
            if not isinstance(url, str) or not re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]*={0,2}", url):
                raise ValueError("Use a PNG, JPEG, or WebP base64 image data URL; remote URLs are unsupported.")
            try:
                if not base64.b64decode(url.split(",", 1)[1], validate=True):
                    raise ValueError()
            except ValueError:
                raise ValueError("Invalid or empty image base64.") from None
        else:
            raise ValueError("Only text and image_url content parts are supported.")
    if sorted(kinds) != ["image_url", "text"]:
        raise ValueError("Supply exactly one text and one image part.")
    return {**payload, "temperature": 0, "top_p": top_p, "max_tokens": tokens, "stream": payload.get("stream", False)}

async def read_payload(request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            raise OverflowError("Request exceeds 128 MiB limit.")
    try:
        return validate(json.loads(body))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Expected a JSON object.") from None


def validate_completion_metadata(metadata, request, output_chars):
    """Backend facts are mandatory; malformed facts are a backend failure, never inferred success."""
    usage = metadata.get("usage")
    reason = metadata.get("finish_reason")
    if reason not in {"stop", "length"} or not isinstance(usage, dict):
        raise RuntimeError("Missing or invalid completion metadata")
    values = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
    if not all(type(value) is int and value >= 0 for value in values):
        raise RuntimeError("Invalid backend token counts")
    prompt, completion, total = values
    limit = request.get("max_tokens", 12384)
    if prompt <= 0 or completion <= 0 or total != prompt + completion or completion > limit:
        raise RuntimeError("Inconsistent backend token accounting")
    if reason == "length" and completion != limit:
        raise RuntimeError("Length finish must exhaust requested output allowance")
    return metadata
