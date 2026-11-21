"""
Cited-answer guard for ResearchRAG (installed by apply_rag_fixes.py).

Wraps /api/query: fuzzy name normalization before retrieval, and a
context-budgeted, citation-enforced answer when the original answer is
empty/uncited (or always, if ANSWER_GUARD_MODE=always).
"""
import asyncio
import json
import os
import re

import requests

from research_portfolio.llm_endpoints import CHAT_URL
from research_portfolio.llm_client import post_chat
MODEL = os.getenv("CHAT_MODEL", "gemma4:12b")
NUM_CTX = int(os.getenv("CHAT_NUM_CTX", "55000"))
NUM_PREDICT = int(os.getenv("CHAT_NUM_PREDICT", "5000"))
MAX_SOURCE_CHARS = int(os.getenv("CHAT_MAX_SOURCE_CHARS", "6000"))
MODE = os.getenv("ANSWER_GUARD_MODE", "fallback").lower()   # fallback | always
CHARS_PER_TOKEN = 3.5
PROMPT_OVERHEAD_TOKENS = 1000

SYSTEM = (
    "You answer questions about UTRGV faculty research using ONLY the numbered evidence.\n"
    "Rules:\n"
    "- Every factual sentence or bullet MUST end with one or more citations like [1] or [2][4].\n"
    "- Only cite numbers that exist in the evidence list.\n"
    "- If the evidence is partial, answer with what it supports and say what is missing.\n"
    "- Only if NO evidence is relevant, reply exactly: INSUFFICIENT_EVIDENCE\n"
    "- Do not list the sources at the end; the application shows them."
)
CITE_RE = re.compile(r"\[(\d+)\]")

PERSON_RESEARCH_RE = re.compile(
    r"\b(research interests?|research areas?|research themes?|"
    r"research topics?|expertise|speciali[sz](?:e|es|ed|ation)|"
    r"what (?:does|is|are).{0,60}research)\b",
    re.IGNORECASE,
)

PROFILE_TYPES = {"interests", "bio"}
DEMONSTRATED_RESEARCH_TYPES = {"publication", "grant", "presentation"}

def _text(s):
    p = s.get("payload") or {}
    return s.get("text") or s.get("chunk_text") or p.get("text") or p.get("chunk_text") or ""


def _source_block(i, s):
    meta = [s.get("chunk_type"), s.get("title"), s.get("year"), s.get("status"),
            s.get("sponsor")]
    head = " | ".join(str(m) for m in meta if m)
    return f"[{i}] {head}\n{_text(s)[:MAX_SOURCE_CHARS]}"

def _chunk_type(source):
    payload = source.get("payload") or {}
    return str(
        source.get("chunk_type")
        or payload.get("chunk_type")
        or ""
    ).strip().lower()


def _question_instruction(question, sources):
    if not PERSON_RESEARCH_RE.search(question or ""):
        return ""

    available = {_chunk_type(source) for source in sources}
    has_profile = bool(available & PROFILE_TYPES)
    has_research = bool(available & DEMONSTRATED_RESEARCH_TYPES)

    instructions = [
        "This question concerns a person's research, expertise, topics, or interests.",
        "Structure the answer using these evidence categories:",
        "1. Explicitly stated interests: use only interests or bio evidence.",
        "2. Demonstrated research themes: synthesize recurring topics supported by "
        "publications, grants, or presentations.",
        "Do not present inferred or demonstrated themes as explicitly stated interests.",
        "Cite representative evidence for every theme.",
    ]

    if has_profile and has_research:
        instructions.append(
            "Both profile evidence and demonstrated-research evidence are available, "
            "so the answer MUST discuss both categories."
        )
    elif has_profile:
        instructions.append(
            "Only profile evidence is available. State that demonstrated research evidence "
            "was not available in the retrieved sources."
        )
    elif has_research:
        instructions.append(
            "Only demonstrated-research evidence is available. State that no explicit "
            "research-interest statement was found in the retrieved sources."
        )

    return "\n".join(instructions)

def build_messages(question, sources):
    budget = (NUM_CTX - NUM_PREDICT - PROMPT_OVERHEAD_TOKENS) * CHARS_PER_TOKEN
    blocks, used = [], 0
    for i, s in enumerate(sources, 1):
        b = _source_block(i, s)
        if used + len(b) > budget:
            break
        blocks.append(b)
        used += len(b)
    
    special_instruction = _question_instruction(question, sources[:len(blocks)])

    user = (
        "Evidence:\n\n"
        + "\n\n".join(blocks)
        + (
            f"\n\nQuestion-specific instructions:\n{special_instruction}"
            if special_instruction else ""
        )
        + f"\n\nQuestion: {question}\nAnswer with citations:"
    )
    
    
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}], len(blocks)


def _chat(messages, think_off=True):
    payload = {"model": MODEL, "messages": messages, "stream": False,
               "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
                           "temperature": 0.2}}
    if think_off:
        payload["think"] = False
    r = post_chat(CHAT_URL, json=payload, timeout=600)
    if r.status_code == 400 and think_off and "think" in r.text.lower():
        return _chat(messages, think_off=False)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}).get("content") or "").strip(), data


def _clean_citations(answer, n):
    out = CITE_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", out)


def generate_cited_answer(question, sources):
    if not sources:
        return {"answer": "", "status": "no_evidence", "used_sources": 0,
                "error": "Retrieval returned no evidence for this question/filters."}
    try:
        messages, n_used = build_messages(question, sources)
        answer, raw = _chat(messages)
        if not answer:
            messages, n_used = build_messages(question, sources[:max(3, n_used // 2)])
            answer, raw = _chat(messages)
        if not answer:
            return {"answer": "", "status": "empty", "used_sources": n_used,
                    "error": (f"Model returned empty content (done_reason="
                              f"{raw.get('done_reason')}, prompt_tokens="
                              f"{raw.get('prompt_eval_count')}, num_ctx={NUM_CTX}).")}
        if answer.strip() == "INSUFFICIENT_EVIDENCE":
            return {"answer": "The retrieved evidence does not answer this question. "
                              "Try a List mode or broaden the filters.",
                    "status": "insufficient", "error": None, "used_sources": n_used}
        answer = _clean_citations(answer, n_used)
        status = "ok"
        if not CITE_RE.search(answer):
            fixed, _ = _chat(messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Rewrite the answer adding [n] citations from "
                                            "the evidence to every factual sentence."}])
            fixed = _clean_citations(fixed, n_used)
            if fixed and CITE_RE.search(fixed):
                answer, status = fixed, "repaired"
            else:
                status = "uncited"
        return {"answer": answer, "status": status, "error": None, "used_sources": n_used}
    except Exception as e:
        return {"answer": "", "status": "error", "used_sources": 0,
                "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------ endpoint wrapper
def _find_req(args, kwargs):
    from research_portfolio.person_match import Q_FIELDS
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, dict) and any(k in v for k in Q_FIELDS):
            return v
        if any(hasattr(v, k) for k in Q_FIELDS) and not isinstance(v, (str, bytes)):
            return v
    return None


def _pre(args, kwargs):
    from research_portfolio.person_match import normalize_query_request
    req = _find_req(args, kwargs)
    if req is None:
        return {"question": None, "notes": [], "clarify": None}
    try:
        return normalize_query_request(req)
    except Exception as e:
        return {"question": None, "notes": [f"Name matching error: {e}"], "clarify": None}


def _clarify_response(info):
    c = info["clarify"]
    c["candidates"] = [x for x in c["candidates"] if x["score"] >= 60][:5]
    names = ", ".join(x["name"] for x in c["candidates"]) or "no close matches"
    return {"kind": "answer", "answer_status": "clarify", "answer_error": None,
            "answer": f'No confident faculty match for "{c["term"]}". Did you mean: {names}?',
            "clarify": c, "name_notes": info["notes"], "sources": []}


def _as_dict(result):
    if isinstance(result, dict):
        return result
    body = getattr(result, "body", None)
    if body is not None:
        try:
            d = json.loads(body)
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def _post(result, info):
    data = _as_dict(result)
    if data is None:
        return result
    data["name_notes"] = info["notes"]
    if "answer" in data and isinstance(data.get("sources"), list):
        ans = data.get("answer") or ""
        needs = MODE == "always" or not ans.strip() or not CITE_RE.search(ans)
        if needs and data["sources"]:
            q = info["question"] or data.get("question") or ""
            res = generate_cited_answer(q, data["sources"])
            if res["answer"]:
                data["answer"] = res["answer"]
            data["answer_status"] = res["status"]
            data["answer_error"] = res["error"]
        else:
            data.setdefault("answer_status", "ok" if ans.strip() else "no_evidence")
            data.setdefault("answer_error", None)
    return data


def guard_query_sync(impl, *args, **kwargs):
    info = _pre(args, kwargs)
    if info["clarify"]:
        return _clarify_response(info)
    return _post(impl(*args, **kwargs), info)


async def guard_query_async(impl, *args, **kwargs):
    info = await asyncio.to_thread(_pre, args, kwargs)
    if info["clarify"]:
        return _clarify_response(info)
    result = impl(*args, **kwargs)
    if asyncio.iscoroutine(result):
        result = await result
    return await asyncio.to_thread(_post, result, info)
