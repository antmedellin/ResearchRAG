"""Chat transport adapter.

Callers build Ollama /api/chat payloads and read Ollama-shaped responses.
CHAT_API=ollama -> sent unchanged.
CHAT_API=openai -> translated to /v1/chat/completions (llama-swap, vLLM) and the
reply (JSON or SSE stream) is translated back to Ollama shape.
"""
import json as _json
import requests
from research_portfolio.llm_endpoints import CHAT_API

_PASS_OPTS = ("temperature", "top_p", "seed", "stop",
              "presence_penalty", "frequency_penalty")


def _to_openai(p):
    opts = p.get("options") or {}
    body = {
        "model": p.get("model"),
        "messages": [{k: v for k, v in m.items() if k in ("role", "content", "name")}
                     for m in p.get("messages", [])],
    }
    if opts.get("num_predict") not in (None, -1):
        body["max_tokens"] = int(opts["num_predict"])
    for k in _PASS_OPTS:
        if opts.get(k) is not None:
            body[k] = opts[k]
    fmt = p.get("format")
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "output", "schema": fmt}}
    think = p.get("think")
    if think is False:
        body["reasoning_effort"] = "none"
    elif isinstance(think, str):
        body["reasoning_effort"] = think
    if p.get("stream", True):          # Ollama's default is stream=true
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _from_openai(d):
    ch = (d.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    u = d.get("usage") or {}
    msg = {"role": m.get("role", "assistant"), "content": m.get("content") or ""}
    r = m.get("reasoning") or m.get("reasoning_content")
    if r:
        msg["thinking"] = r
    return {"model": d.get("model"), "message": msg, "done": True,
            "done_reason": ch.get("finish_reason") or "stop",
            "prompt_eval_count": u.get("prompt_tokens"),
            "eval_count": u.get("completion_tokens")}


class _OllamaShaped:
    def __init__(self, resp, streaming):
        self._r, self._streaming = resp, streaming
        self.status_code, self.ok = resp.status_code, resp.ok
        self.headers, self.url, self.reason = resp.headers, resp.url, resp.reason

    def raise_for_status(self):
        return self._r.raise_for_status()

    def _chunks(self):
        model, usage, reason = None, {}, "stop"
        for raw in self._r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = _json.loads(data)
            model = d.get("model", model)
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                reason = ch.get("finish_reason") or reason
                delta = ch.get("delta") or {}
                c = delta.get("content") or ""
                t = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if c or t:
                    msg = {"role": "assistant", "content": c}
                    if t:
                        msg["thinking"] = t
                    yield {"model": model, "message": msg, "done": False}
        yield {"model": model, "message": {"role": "assistant", "content": ""},
               "done": True, "done_reason": reason,
               "prompt_eval_count": usage.get("prompt_tokens"),
               "eval_count": usage.get("completion_tokens")}

    def json(self, **kw):
        if not self.ok:
            return self._r.json(**kw)
        if not self._streaming:
            return _from_openai(self._r.json(**kw))
        content, thinking, last = [], [], {}
        for c in self._chunks():
            content.append(c["message"].get("content", ""))
            thinking.append(c["message"].get("thinking", ""))
            last = c
        last["message"] = {"role": "assistant", "content": "".join(content)}
        if any(thinking):
            last["message"]["thinking"] = "".join(thinking)
        return last

    def iter_lines(self, chunk_size=512, decode_unicode=False, delimiter=None):
        if not self.ok:
            yield from self._r.iter_lines(chunk_size, decode_unicode, delimiter)
            return
        items = self._chunks() if self._streaming else [self.json()]
        for c in items:
            s = _json.dumps(c)
            yield s if decode_unicode else s.encode()

    @property
    def text(self):
        if not self.ok:
            return self._r.text
        return _json.dumps(self.json())

    def close(self):
        self._r.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def post_chat(url, json=None, timeout=None, stream=False, **kwargs):
    if CHAT_API != "openai":
        return requests.post(url, json=json, timeout=timeout, stream=stream, **kwargs)
    body = _to_openai(json or {})
    want = bool(body.get("stream"))
    r = requests.post(url, json=body, timeout=timeout, stream=stream or want, **kwargs)
    return _OllamaShaped(r, want)


# ---------------- embeddings ----------------
from research_portfolio.llm_endpoints import EMBED_API


class _OllamaEmbedShaped:
    def __init__(self, resp):
        self._r = resp
        self.status_code, self.ok = resp.status_code, resp.ok
        self.headers, self.url, self.reason = resp.headers, resp.url, resp.reason

    def raise_for_status(self):
        return self._r.raise_for_status()

    def json(self, **kw):
        d = self._r.json(**kw)
        if not self.ok:
            return d
        data = sorted(d.get("data") or [], key=lambda x: x.get("index", 0))
        u = d.get("usage") or {}
        return {"model": d.get("model"),
                "embeddings": [x["embedding"] for x in data],
                "prompt_eval_count": u.get("prompt_tokens")}

    @property
    def text(self):
        return _json.dumps(self.json()) if self.ok else self._r.text

    def close(self):
        self._r.close()


def post_embed(url, json=None, timeout=None, **kwargs):
    if EMBED_API != "openai":
        return requests.post(url, json=json, timeout=timeout, **kwargs)
    p = json or {}
    body = {"model": p.get("model"), "input": p.get("input", p.get("prompt", ""))}
    if p.get("dimensions"):
        body["dimensions"] = p["dimensions"]
    return _OllamaEmbedShaped(requests.post(url, json=body, timeout=timeout, **kwargs))
