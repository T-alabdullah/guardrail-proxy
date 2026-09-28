"""guardrail-proxy: an OpenAI-compatible /v1/chat/completions endpoint in front of Ollama.

optional Llama Guard check, switched on per request with "guardrails": true.
- The input is checked before the model runs (streaming and non-streaming).
- The output is checked after generation (non-streaming only, for now).
- A blocked request returns 200 with a refusal, finish_reason "content_filter",
  and a "guardrail" field naming the Llama Guard category.
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import guard

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# CPU inference is slow, so the read timeout is generous. Connect should fail fast.
TIMEOUT = httpx.Timeout(300.0, connect=5.0)
# If Llama Guard itself fails: "closed" = refuse the request (503), "open" = let it through unchecked.
GUARD_FAIL_MODE = os.getenv("GUARD_FAIL_MODE", "closed")
log = logging.getLogger("uvicorn.error")  # shows up in the uvicorn terminal

# OpenAI parameter name -> Ollama "options" name
PARAM_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "max_tokens": "num_predict",
    "max_completion_tokens": "num_predict",  # newer name for the same thing
}
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared HTTP client for the lifetime of the service (reuses connections).
    app.state.ollama = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=TIMEOUT)
    yield
    await app.state.ollama.aclose()


app = FastAPI(title="guardrail-proxy", lifespan=lifespan)


# ---------- helpers: errors ----------

def openai_error(status: int, message: str, err_type: str = "invalid_request_error",
                 code: str | None = None) -> JSONResponse:
    """Errors in the shape the OpenAI client expects, so it raises the right exception."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


def error_from_ollama(r: httpx.Response) -> JSONResponse:
    try:
        detail = r.json().get("error", r.text)
    except ValueError:
        detail = r.text
    if r.status_code == 404:
        return openai_error(404, detail, code="model_not_found")
    return openai_error(502, f"Ollama error: {detail}", "api_error")


# ---------- helpers: translation ----------

def flatten_content(content) -> str:
    """OpenAI allows content as a string or a list of parts. Ollama wants a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def normalize_messages(raw_messages: list) -> list[dict]:
    """OpenAI messages -> plain {role, content} dicts with string content."""
    messages = []
    for m in raw_messages:
        role = m.get("role", "user")
        if role == "developer":  # newer OpenAI name for the system role
            role = "system"
        messages.append({"role": role, "content": flatten_content(m.get("content"))})
    return messages


def to_ollama_request(body: dict, messages: list[dict], stream: bool) -> dict:
    options = {}
    for openai_name, ollama_name in PARAM_MAP.items():
        if body.get(openai_name) is not None:
            options[ollama_name] = body[openai_name]
    stop = body.get("stop")
    if stop is not None:
        options["stop"] = [stop] if isinstance(stop, str) else stop
    return {"model": body["model"], "messages": messages, "stream": stream, "options": options}


def finish_reason_from(data: dict) -> str:
    return "length" if data.get("done_reason") == "length" else "stop"


def usage_from(data: dict) -> dict:
    p = data.get("prompt_eval_count", 0)
    c = data.get("eval_count", 0)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def wants_usage(body: dict) -> bool:
    return bool((body.get("stream_options") or {}).get("include_usage"))


def completion(model: str, content: str, finish_reason: str, usage: dict,
               guardrail: dict | None = None) -> dict:
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }
    if guardrail is not None:
        resp["guardrail"] = guardrail
    return resp


def make_chunker(model: str):
    """Returns a function that builds OpenAI stream chunks sharing one id."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    return chunk


def usage_chunk(chunk, usage: dict) -> dict:
    c = chunk({})
    c["choices"] = []
    c["usage"] = usage
    return c


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# ---------- helpers: timings ----------

def timings_ms(data: dict) -> dict:
    """Ollama reports durations in nanoseconds."""
    return {k: round(data.get(f"{k}_duration", 0) / 1e6, 1) for k in ("load", "prompt_eval", "eval", "total")}


def timing_headers(data: dict) -> dict:
    """Non-streaming only: the OpenAI client ignores unknown headers."""
    return {f"x-ollama-{k.replace('_', '-')}-ms": str(v) for k, v in timings_ms(data).items()}


def log_timings(data: dict, stream: bool) -> None:
    t = timings_ms(data)
    log.info("ollama %s: load=%.1fms prompt_eval=%.1fms (%d tok) eval=%.1fms (%d tok) total=%.1fms",
             "stream" if stream else "non-stream", t["load"], t["prompt_eval"],
             data.get("prompt_eval_count", 0), t["eval"], data.get("eval_count", 0), t["total"])


# ---------- helpers: guardrails ----------

def refusal_text(verdict: guard.Verdict, stage: str) -> str:
    what = "request" if stage == "input" else "response"
    return f"Sorry, I can't help with that. The {what} was blocked by the content guardrail ({verdict.describe()})."


def guard_info(blocked: guard.Verdict | None = None, stage: str | None = None) -> dict:
    """The extra 'guardrail' field the caller sees."""
    if blocked is None:
        return {"blocked": False, "stage": None, "categories": []}
    return {"blocked": True, "stage": stage, "categories": blocked.categories}


def guard_headers(verdicts: list) -> dict:
    if not verdicts:
        return {}
    blocked = [v for v in verdicts if not v.safe]
    return {
        "x-guard-ms": f"{sum(v.elapsed_ms for v in verdicts):.1f}",
        "x-guardrail-blocked": "true" if blocked else "false",
        "x-guardrail-categories": ",".join(c for v in blocked for c in v.categories),
    }


async def run_guard(request: Request, messages: list[dict], stage: str):
    """Returns (verdict, error_response). Both are None if the guard failed and GUARD_FAIL_MODE=open."""
    try:
        verdict = await guard.classify(request.app.state.ollama, messages)
    except guard.GuardError as e:
        log.error("guard %s check failed: %s", stage, e)
        if GUARD_FAIL_MODE == "open":
            log.warning("GUARD_FAIL_MODE=open: continuing without a %s check", stage)
            return None, None
        return None, openai_error(503, "The content guardrail is unavailable, so the request was not processed.",
                                  "api_error", code="guardrail_unavailable")
    log.info("guard %s: %s (%.0f ms)", stage, verdict.summary(), verdict.elapsed_ms)
    return verdict, None


def blocked_input_response(body: dict, verdict: guard.Verdict, stream: bool):
    """The model never ran. The refusal is a normal completion, so every client can display it."""
    model = body["model"]
    text = refusal_text(verdict, "input")
    info = guard_info(verdict, "input")
    headers = guard_headers([verdict])
    if not stream:
        return JSONResponse(completion(model, text, "content_filter", ZERO_USAGE, info), headers=headers)
    # The caller asked for a stream, so the refusal has to be a stream too, or the client breaks.
    return StreamingResponse(
        blocked_events(model, text, info, wants_usage(body)),
        media_type="text/event-stream",
        headers={**headers, "Cache-Control": "no-cache"},
    )


async def blocked_events(model: str, text: str, info: dict, include_usage: bool):
    chunk = make_chunker(model)
    yield sse(chunk({"role": "assistant", "content": ""}))
    yield sse(chunk({"content": text}))
    final = chunk({}, "content_filter")
    final["guardrail"] = info
    yield sse(final)
    if include_usage:
        yield sse(usage_chunk(chunk, ZERO_USAGE))
    yield "data: [DONE]\n\n"


# ---------- endpoints ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return openai_error(400, "Request body must be valid JSON.")

    if not body.get("model") or not body.get("messages"):
        return openai_error(400, "'model' and 'messages' are required.")
    if body.get("n", 1) != 1:
        return openai_error(400, "Only n=1 is supported.")

    guardrails = body.get("guardrails")
    if guardrails is None:
        guardrails = False
    if not isinstance(guardrails, bool):
        return openai_error(400, "'guardrails' must be true or false.")

    messages = normalize_messages(body["messages"])
    stream = bool(body.get("stream"))

    input_verdict = None
    if guardrails:
        input_verdict, err = await run_guard(request, messages, "input")
        if err:
            return err
        if input_verdict and not input_verdict.safe:
            return blocked_input_response(body, input_verdict, stream)

    if stream:
        return await stream_completion(request, body, messages, guardrails, input_verdict)
    return await non_stream_completion(request, body, messages, guardrails, input_verdict)


async def non_stream_completion(request: Request, body: dict, messages: list[dict],
                                guardrails: bool, input_verdict):
    try:
        r = await request.app.state.ollama.post("/api/chat", json=to_ollama_request(body, messages, stream=False))
    except httpx.RequestError as e:
        return openai_error(502, f"Could not reach Ollama at {OLLAMA_URL}: {e}", "api_error")
    if r.status_code != 200:
        return error_from_ollama(r)

    data = r.json()
    log_timings(data, stream=False)
    model = body["model"]
    content = data.get("message", {}).get("content", "")
    headers = timing_headers(data)
    verdicts = [input_verdict] if input_verdict else []

    if guardrails:
        # Output check: the same conversation plus the model's answer as the last (assistant) turn.
        out_verdict, err = await run_guard(request, messages + [{"role": "assistant", "content": content}], "output")
        if err:
            return err
        if out_verdict:
            verdicts.append(out_verdict)
            if not out_verdict.safe:
                headers.update(guard_headers(verdicts))
                # The model did run, so the usage is real even though the answer is withheld.
                return JSONResponse(
                    completion(model, refusal_text(out_verdict, "output"), "content_filter",
                               usage_from(data), guard_info(out_verdict, "output")),
                    headers=headers,
                )

    headers.update(guard_headers(verdicts))
    return JSONResponse(
        completion(model, content, finish_reason_from(data), usage_from(data),
                   guard_info() if guardrails else None),
        headers=headers,
    )


async def stream_completion(request: Request, body: dict, messages: list[dict],
                            guardrails: bool, input_verdict):
    client = request.app.state.ollama
    req = client.build_request("POST", "/api/chat", json=to_ollama_request(body, messages, stream=True))
    try:
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as e:
        return openai_error(502, f"Could not reach Ollama at {OLLAMA_URL}: {e}", "api_error")

    # Nothing has been sent to the caller yet, so we can still return a proper HTTP error.
    if upstream.status_code != 200:
        await upstream.aread()
        await upstream.aclose()
        return error_from_ollama(upstream)

    headers = {"Cache-Control": "no-cache", **guard_headers([input_verdict] if input_verdict else [])}
    return StreamingResponse(
        sse_events(upstream, body["model"], wants_usage(body), guardrails),
        media_type="text/event-stream",
        headers=headers,
    )


async def sse_events(upstream: httpx.Response, model: str, include_usage: bool, guardrails: bool):
    """Translate Ollama's NDJSON stream into OpenAI's SSE chunk format."""
    chunk = make_chunker(model)
    try:
        # First chunk announces the role, as OpenAI does.
        yield sse(chunk({"role": "assistant", "content": ""}))

        async for line in upstream.aiter_lines():
            if not line.strip():
                continue
            data = json.loads(line)

            if "error" in data:
                # Status 200 has already been sent; the only place left for the error is the stream.
                yield sse({"error": {"message": data["error"], "type": "api_error", "param": None, "code": None}})
                break

            text = data.get("message", {}).get("content", "")
            if text:
                yield sse(chunk({"content": text}))

            if data.get("done"):
                log_timings(data, stream=True)
                final = chunk({}, finish_reason_from(data))
                if guardrails:
                    final["guardrail"] = guard_info()
                    log.warning("stream: output was NOT checked by the guardrail (step 4c)")
                yield sse(final)
                if include_usage:
                    yield sse(usage_chunk(chunk, usage_from(data)))
                break

        yield "data: [DONE]\n\n"
    finally:
        # Also runs if the caller disconnects mid-stream: closing the upstream
        # connection makes Ollama stop generating for nobody.
        await upstream.aclose()


@app.get("/v1/models")
async def list_models(request: Request):
    """Some OpenAI-compatible tools call this first to see which models exist."""
    try:
        r = await request.app.state.ollama.get("/api/tags")
        r.raise_for_status()
    except httpx.HTTPError as e:
        return openai_error(502, f"Could not list models from Ollama: {e}", "api_error")
    models = r.json().get("models", [])
    return {
        "object": "list",
        "data": [{"id": m["name"], "object": "model", "created": 0, "owned_by": "ollama"} for m in models],
    }