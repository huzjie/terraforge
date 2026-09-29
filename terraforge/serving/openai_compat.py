"""OpenAI-compatible request/response shaping."""
import json
import time
import uuid


def chat_response(model, content, prompt_tokens=0, completion_tokens=0):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def embedding_response(model, embedding):
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": embedding}],
        "model": model,
    }


def models_response(models):
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": int(time.time()),
                  "owned_by": "terraforge"} for m in models],
    }


def parse_chat_request(body):
    """Return (model, messages_text)."""
    model = body.get("model", "terraforge-spatial-9b")
    messages = body.get("messages", [])
    text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    return model, text
