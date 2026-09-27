"""guardrail-proxy: an OpenAI-compatible /v1/chat/completions endpoint in front of Ollama.

Step 2: non-streaming proxy. Streaming and guardrails come later.
"""
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# CPU inference is slow, so the read timeout is generous. Connect should fail fast.
TIMEOUT = httpx.Timeout(300.0, connect=5.0)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared HTTP client for the lifetime of the service (reuses connections).
    app.state.ollama = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=TIMEOUT)
    yield
    await app.state.ollama.aclose()


app = FastAPI(title="guardrail-proxy", lifespan=lifespan)


# ---------- helpers ----------

def openai_error(status: int, message: str, err_type: str = "invalid_request_error",
                 code: str | None = None) -> JSONResponse:
    """Errors in the shape the OpenAI client expects, so it raises the right exception."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


def flatten_content(content) -> str:
    """OpenAI allows content as a string or a list of parts. Ollama wants a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def to_ollama_request(body: dict) -> dict:
    messages = []
    for m in body["messages"]:
        role = m.get("role", "user")
        if role == "developer":  # newer OpenAI name for the system role
            role = "system"
        messages.append({"role": role, "content": flatten_content(m.get("content"))})

    options = {}
    for openai_name, ollama_name in PARAM_MAP.items():
        if body.get(openai_name) is not None:
            options[ollama_name] = body[openai_name]
    stop = body.get("stop")
    if stop is not None:
        options["stop"] = [stop] if isinstance(stop, str) else stop

    return {"model": body["model"], "messages": messages, "stream": False, "options": options}


def to_openai_response(data: dict, model: str) -> dict:
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    finish_reason = "length" if data.get("done_reason") == "length" else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": data.get("message", {}).get("content", "")},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def timing_headers(data: dict) -> dict:
    """Ollama's timings (nanoseconds) as response headers, in milliseconds.
    The OpenAI client ignores unknown headers, so this doesn't affect compatibility."""
    ms = lambda key: f"{data.get(key, 0) / 1e6:.1f}"
    return {
        "x-ollama-load-ms": ms("load_duration"),
        "x-ollama-prompt-eval-ms": ms("prompt_eval_duration"),
        "x-ollama-eval-ms": ms("eval_duration"),
        "x-ollama-total-ms": ms("total_duration"),
    }


# ---------- endpoints ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return openai_error(400, "Request body must be valid JSON.")

    if not body.get("model") or not body.get("messages"):
        return openai_error(400, "'model' and 'messages' are required.")
    if body.get("stream"):
        return openai_error(400, "Streaming is not supported yet.", code="stream_not_supported")
    if body.get("n", 1) != 1:
        return openai_error(400, "Only n=1 is supported.")

    try:
        r = await request.app.state.ollama.post("/api/chat", json=to_ollama_request(body))
    except httpx.RequestError as e:
        return openai_error(502, f"Could not reach Ollama at {OLLAMA_URL}: {e}", "api_error")

    if r.status_code != 200:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        if r.status_code == 404:
            return openai_error(404, detail, code="model_not_found")
        return openai_error(502, f"Ollama error: {detail}", "api_error")

    data = r.json()
    return JSONResponse(to_openai_response(data, body["model"]), headers=timing_headers(data))


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
