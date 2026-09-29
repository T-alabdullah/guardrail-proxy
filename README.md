# guardrail-proxy

A small service that sits in front of a local language model (via Ollama) and decides whether
requests are allowed through. It exposes an **OpenAI-compatible** `POST /v1/chat/completions`
endpoint: the official OpenAI Python client works against it with nothing changed but the base URL.

With `"guardrails": true` in the request body, the prompt is checked by **Llama Guard 3** first.
Unsafe prompts never reach the model; the caller gets a refusal instead.

## What it does

- OpenAI chat completions format in, Ollama's native API out, OpenAI format back
- Streaming (Server-Sent Events) and non-streaming
- Optional guardrail per request (`"guardrails": true`):
  - **Input check** before the model runs (streaming and non-streaming)
  - **Output check** after generation (non-streaming)
  - **Streaming output check** in the background while tokens are sent; an unsafe verdict cuts the stream
- Blocked requests return **HTTP 200** with a refusal, `finish_reason: "content_filter"`,
  and a `guardrail` field naming the Llama Guard category (e.g. `S2`)
- Ollama's timings (model load, prompt processing, generation) as response headers

Design decisions and their reasoning: [DECISIONS.md](DECISIONS.md).
Measurements: [MEASUREMENTS.md](MEASUREMENTS.md). Background: [UNDERSTANDING.md](UNDERSTANDING.md).

## Requirements

- macOS, Linux, or Windows with WSL2
- [Ollama](https://ollama.com)
- Python 3.11 (conda or venv)
- About **5 GB of free RAM** for both models (4.4 GB when both are loaded)
- CPU is enough; no GPU needed

Tested on: MacBook Pro 2019, Intel i9-9880H, 16 GB RAM, CPU only, Ollama 0.33.2, Python 3.11, openai 3.19.2.

## Setup

### 1. Install Ollama and pull the models

macOS: install the app from [ollama.com](https://ollama.com) or run `brew install ollama`.
Linux / WSL2: `curl -fsSL https://ollama.com/install.sh | sh`

```bash
ollama pull llama3.2:3b        # the chat model
ollama pull llama-guard3:1b    # the guardrail model
```

### 2. Start Ollama with keep-alive (terminal 1)

```bash
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

`OLLAMA_KEEP_ALIVE=-1` keeps both models in memory permanently. Without it, Ollama unloads a model after
5 minutes idle, and the next request pays 4-7 s of load time. See [DECISIONS.md](DECISIONS.md) for the trade-off.

If you get `address already in use`, Ollama is already running (often the menu-bar app). Quit it first
(`osascript -e 'quit app "Ollama"'` on macOS), then run the command again.

### 3. Python environment

```bash
git clone <this repo> && cd guardrail-proxy

# conda:
conda create -n guardproxy python=3.11 -y
conda activate guardproxy
# or venv:
# python3.11 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Start the proxy (terminal 2)

```bash
uvicorn app:app --port 8000
```

Add `--reload` during development to restart automatically when a file changes.

### 5. Check it works (terminal 3)

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Say hello in five words."}]}'
```

The first request takes a few seconds longer while the model loads.

## Usage

### With the OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(resp.choices[0].message.content)
```

The client requires some `api_key`; the proxy currently ignores it.

### Turning the guardrail on

`guardrails` is not a standard OpenAI parameter, so the Python client sends it via `extra_body`:

```python
resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "How do I hotwire a car?"}],
    extra_body={"guardrails": True},
)
print(resp.choices[0].finish_reason)        # content_filter
print(getattr(resp, "guardrail", None))     # {'blocked': True, 'stage': 'input', 'categories': ['S2']}
```

With curl, it's a normal field in the body: `"guardrails": true`.

### Streaming

```python
stream = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Explain why the sea is salty."}],
    stream=True,
    stream_options={"include_usage": True},
    extra_body={"guardrails": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## API reference

### `POST /v1/chat/completions`

Supported request fields: `model`, `messages` (string content or text parts; the `developer` role is
treated as `system`), `stream`, `stream_options.include_usage`, `temperature`, `top_p`, `max_tokens`,
`max_completion_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, and `guardrails`.

Not supported: `n` > 1 (returns 400). Tools / function calling, images, and `response_format` are
not implemented and are ignored.

### `GET /v1/models`

Lists the models available in Ollama.

### The `guardrails` flag

| Value | Behaviour |
|---|---|
| missing, `null`, or `false` | Straight through to the model, no checks |
| `true` | Llama Guard checks the input, then the output (see below) |
| anything else (e.g. `"yes"`) | `400 Bad Request`, so a typo never silently means "no guardrail" |

With `true`:

- **Input unsafe:** the model never runs. Response is a normal completion with a refusal text,
  `finish_reason: "content_filter"`, usage 0, and `guardrail: {blocked: true, stage: "input", categories: [...]}`.
  If the caller asked for a stream, the refusal arrives as a stream.
- **Output unsafe, non-streaming:** the answer is withheld and replaced by a refusal (`stage: "output"`).
- **Output unsafe, streaming:** tokens are sent immediately and checked in the background. On an unsafe
  verdict the stream is cut with a notice and `finish_reason: "content_filter"`. Text sent before the cut
  has already reached the caller (the leak window; measured in [MEASUREMENTS.md](MEASUREMENTS.md)).
- **Llama Guard unavailable:** `503` (fail closed), unless `GUARD_FAIL_MODE=open`.

Example of a blocked request (non-streaming):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "llama3.2:3b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant",
                "content": "Sorry, I can't help with that. The request was blocked by the content guardrail (S2: Non-Violent Crimes)."},
    "finish_reason": "content_filter"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "guardrail": {"blocked": true, "stage": "input", "categories": ["S2"]}
}
```

### Response headers (non-streaming)

| Header | Meaning |
|---|---|
| `x-ollama-load-ms` | Model load time (near 0 when already loaded) |
| `x-ollama-prompt-eval-ms` | Prompt processing |
| `x-ollama-eval-ms` | Generation |
| `x-ollama-total-ms` | Total inside Ollama |
| `x-guard-ms` | Time spent in Llama Guard (input + output) |
| `x-guardrail-blocked`, `x-guardrail-categories` | Guardrail verdict |

For streaming, headers are sent before the timings exist, so timings are written to the proxy's log instead.

## Configuration

Proxy (environment variables when starting uvicorn):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama runs |
| `GUARD_MODEL` | `llama-guard3:1b` | Guardrail model |
| `GUARD_FAIL_MODE` | `closed` | `closed`: refuse (503) if Llama Guard fails. `open`: let the request through unchecked |
| `STREAM_GUARD_MIN_CHARS` | `100` | Streaming: unchecked characters before the next background check. Smaller = smaller leak window, more CPU load |
| `GUARD_TEST_FAKE_UNSAFE_WORD` | (off) | **Test only.** Outputs containing this word are treated as unsafe (S1), to demo a cut stream |

Ollama (environment variables when starting `ollama serve`):

| Variable | Used here | Meaning |
|---|---|---|
| `OLLAMA_KEEP_ALIVE` | `-1` | How long models stay loaded when idle (default 5m) |
| `OLLAMA_MAX_LOADED_MODELS` | default | Max models in memory. `1` forces the two models to swap on every guarded request |
| `OLLAMA_NUM_PARALLEL` | default (1) | Parallel requests per model |

## Tests

With Ollama and the proxy running:

```bash
python test_client.py      # OpenAI client, non-streaming: normal, max_tokens, unknown model, model list
python test_stream.py      # OpenAI client, streaming: TTFT and throughput
python test_guard.py       # guardrail: safe, blocked, no guard, blocked stream, invalid flag
python test_stream_guard.py  # streaming with background output checks, guard off vs on
```

To see a stream being cut by the output check, start the proxy in test mode, then run `test_stream_guard.py`:

```bash
GUARD_TEST_FAKE_UNSAFE_WORD=twenty uvicorn app:app --port 8000
```

## Reproducing the measurements

Close other heavy apps, keep the machine plugged in, and run the proxy **without** test mode.

```bash
mkdir -p results
python bench.py 2>&1 | tee results/bench-run1.txt              # guard off vs on, stream vs not, warm vs cold (~10 min)
python longprompt.py 2>&1 | tee results/longprompt-run1.txt    # prompt processing vs generation by prompt length (~5 min)
python guard_eval.py 2>&1 | tee results/guard-eval-run1.txt    # Llama Guard on safe / unsafe / ambiguous prompts (~1 min)
```

Raw data is written to `results/*.csv`. Results and interpretation: [MEASUREMENTS.md](MEASUREMENTS.md).

## Repository layout

```
app.py                 the service (FastAPI)
guard.py               Llama Guard client and verdict parsing
test_*.py              functional tests using the OpenAI client
bench.py               latency benchmark
longprompt.py          prompt-length experiment
guard_eval.py          guardrail accuracy test set
results/               raw measurement data and console output
DECISIONS.md           design decisions and reasoning
MEASUREMENTS.md        measurements with the commands that produced them
UNDERSTANDING.md       KV cache, Ollama, batching, platform context
NOTES.md               working notes
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `address already in use` when starting Ollama | Ollama is already running (often the menu-bar app). Quit it, or use the running one |
| `ModuleNotFoundError: No module named 'openai'` | Wrong Python environment. Run `conda activate guardproxy` in every new terminal |
| `Connection refused` from a test | The proxy isn't running. Start it (setup step 4) |
| Requests hang with no error | With `--reload`, a broken `app.py` keeps the port open but serves nothing. Check the uvicorn terminal for a `SyntaxError`, and run `python -c "import app"` |
| First request is very slow | The model is loading (4-7 s). Use `OLLAMA_KEEP_ALIVE=-1` |
| Everything is slow and inconsistent | Memory pressure. Check Activity Monitor; both models need about 4.4 GB |
| `503 guardrail_unavailable` | Llama Guard isn't pulled or failed. Run `ollama pull llama-guard3:1b` |