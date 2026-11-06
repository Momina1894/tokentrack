"""
tokentrack — log every LLM call's exact token usage and cost, locally.

Usage (one extra line in your existing code):

    from openai import OpenAI
    import tokentrack

    client = tokentrack.wrap(OpenAI(), tag="my-script")
    # ...use `client` exactly as before...

Works the same way for Anthropic:

    from anthropic import Anthropic
    client = tokentrack.wrap(Anthropic(), tag="my-script")

Every call is written to tokentrack.db (SQLite) in the current folder.
Set TOKENTRACK_DB=/some/path.db to put it elsewhere.
"""

import functools
import time
import uuid
from typing import Any

from .pricing import cost_usd
from .storage import SQLiteStore

__all__ = ["wrap", "new_conversation"]

_store: SQLiteStore | None = None


def _get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore()
    return _store


def new_conversation() -> str:
    """Make a fresh conversation id. Pass it to every turn of one chat."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# Pull token counts out of each provider's response object.
# This is the only provider-specific knowledge in the whole library.
# --------------------------------------------------------------------------

def _usage_openai(resp: Any) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    # chat.completions: prompt_tokens / completion_tokens
    # responses API:    input_tokens  / output_tokens
    inp = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0) or 0
    cached = 0
    details = getattr(u, "prompt_tokens_details", None) or getattr(u, "input_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {"input_tokens": inp, "output_tokens": out, "cached_tokens": cached}


def _usage_anthropic(resp: Any) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    cached = getattr(u, "cache_read_input_tokens", 0) or 0
    # Anthropic reports uncached input separately; add cached back so
    # input_tokens means "total input" like it does for OpenAI.
    inp = (getattr(u, "input_tokens", 0) or 0) + cached + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    out = getattr(u, "output_tokens", 0) or 0
    return {"input_tokens": inp, "output_tokens": out, "cached_tokens": cached}


def _text_openai(resp: Any) -> str | None:
    try:
        if hasattr(resp, "choices"):
            return resp.choices[0].message.content
        if hasattr(resp, "output_text"):
            return resp.output_text
    except Exception:  # noqa: BLE001
        pass
    return None


def _text_anthropic(resp: Any) -> str | None:
    try:
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    except Exception:  # noqa: BLE001
        return None


def _prompt_text(kwargs: dict) -> str | None:
    """Flatten the messages the user sent into one string (for opt-in storage)."""
    msgs = kwargs.get("messages") or kwargs.get("input")
    if isinstance(msgs, str):
        return msgs
    if isinstance(msgs, list):
        parts = []
        if kwargs.get("system"):
            parts.append(f"[system] {kwargs['system']}")
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                parts.append(f"[{m.get('role', '?')}] {c}")
        return "\n".join(parts)
    return None


# --------------------------------------------------------------------------
# The wrapper itself
# --------------------------------------------------------------------------

def _make_logged(original, provider, usage_fn, text_fn, defaults, store_text):
    @functools.wraps(original)
    def logged(*args, **kwargs):
        # Let the caller override conversation_id / tag per call.
        conversation_id = kwargs.pop("conversation_id", defaults.get("conversation_id"))
        tag = kwargs.pop("tag", defaults.get("tag"))
        t0 = time.perf_counter()
        resp = original(*args, **kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        if kwargs.get("stream"):
            # Streaming: we don't get usage until the stream ends, so we
            # hand back a generator that logs when it's exhausted.
            return _wrap_stream(resp, provider, kwargs, conversation_id, tag, store_text, t0)

        _record(resp, provider, usage_fn, text_fn, kwargs, conversation_id, tag, store_text, latency_ms)
        return resp

    return logged


def _record(resp, provider, usage_fn, text_fn, kwargs, conversation_id, tag, store_text, latency_ms):
    try:
        usage = usage_fn(resp)
        model = getattr(resp, "model", None) or kwargs.get("model", "unknown")
        event = {
            "provider": provider,
            "model": model,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
            "cost_usd": cost_usd(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("cached_tokens", 0)),
            "latency_ms": latency_ms,
            "conversation_id": conversation_id,
            "tag": tag,
            "prompt_text": _prompt_text(kwargs) if store_text else None,
            "response_text": text_fn(resp) if store_text else None,
            "meta": {"requested_model": kwargs.get("model")},
        }
        _get_store().write_event(event)
    except Exception:  # noqa: BLE001
        pass  # never break the caller's program


def _wrap_stream(stream, provider, kwargs, conversation_id, tag, store_text, t0):
    """Yield chunks through, then log once the provider has sent usage."""
    inp = out = cached = 0
    model = kwargs.get("model", "unknown")
    pieces = []
    try:
        for chunk in stream:
            if provider == "openai":
                u = getattr(chunk, "usage", None)
                if u:  # only the final chunk carries usage (needs stream_options={"include_usage": True})
                    inp, out = u.prompt_tokens, u.completion_tokens
                    d = getattr(u, "prompt_tokens_details", None)
                    cached = getattr(d, "cached_tokens", 0) if d else 0
                try:
                    pieces.append(chunk.choices[0].delta.content or "")
                except Exception:  # noqa: BLE001
                    pass
            else:  # anthropic
                et = getattr(chunk, "type", "")
                if et == "message_start":
                    u = chunk.message.usage
                    cached = getattr(u, "cache_read_input_tokens", 0) or 0
                    inp = u.input_tokens + cached + (getattr(u, "cache_creation_input_tokens", 0) or 0)
                    model = chunk.message.model
                elif et == "message_delta":
                    out = chunk.usage.output_tokens
                elif et == "content_block_delta" and getattr(chunk.delta, "type", "") == "text_delta":
                    pieces.append(chunk.delta.text)
            yield chunk
    finally:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _get_store().write_event({
            "provider": provider,
            "model": model,
            "input_tokens": inp,
            "output_tokens": out,
            "cached_tokens": cached,
            "cost_usd": cost_usd(model, inp, out, cached),
            "latency_ms": latency_ms,
            "conversation_id": conversation_id,
            "tag": tag,
            "prompt_text": _prompt_text(kwargs) if store_text else None,
            "response_text": "".join(pieces) if store_text else None,
            "meta": {"stream": True},
        })


def _provider_from_url(base_url: str) -> str:
    """
    Many providers speak the OpenAI API format at their own address.
    Name them properly instead of calling everything 'openai'.
    """
    host = base_url.lower()
    for key, name in [
        ("groq.com", "groq"), ("googleapis.com", "gemini"), ("localhost", "ollama"),
        ("127.0.0.1", "ollama"), ("openrouter.ai", "openrouter"), ("mistral.ai", "mistral"),
        ("deepseek.com", "deepseek"), ("together.xyz", "together"), ("x.ai", "xai"),
    ]:
        if key in host:
            return name
    return "openai"


def wrap(client, *, conversation_id: str | None = None, tag: str | None = None, store_text: bool = False):
    """
    Patch an OpenAI or Anthropic client so every call is logged.

    conversation_id: default id for grouping turns (override per call).
    tag:             default label for this script/app.
    store_text:      ALSO save the prompt and response text. Off by default.
                     Turn on only when you're OK with prompts living in the db
                     (needed later for the AI coach).
    Returns the same client object, so you can use it in place.
    """
    defaults = {"conversation_id": conversation_id, "tag": tag}
    name = type(client).__module__.split(".")[0]

    if name == "openai":
        provider = _provider_from_url(str(getattr(client, "base_url", "")))
        client.chat.completions.create = _make_logged(
            client.chat.completions.create, provider, _usage_openai, _text_openai, defaults, store_text
        )
        if hasattr(client, "responses"):
            client.responses.create = _make_logged(
                client.responses.create, provider, _usage_openai, _text_openai, defaults, store_text
            )
    elif name == "anthropic":
        client.messages.create = _make_logged(
            client.messages.create, "anthropic", _usage_anthropic, _text_anthropic, defaults, store_text
        )
    else:
        raise TypeError(f"tokentrack.wrap: don't know how to wrap a {type(client)!r}")
    return client