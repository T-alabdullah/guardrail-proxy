## Step 1 – setup
- Intel i9-9880H, 16 GB, CPU only
- ollama ps: llama3.2:3b 2.6 GB, llama-guard3:1b 1.8 GB (more than download size: KV cache + buffers)
- Default keep-alive: 5 min
- llama-guard3: "hotwire a car" -> unsafe S2, "bake bread" -> safe

## Step 2 – proxy
- OpenAI client 3.19.2 works with base_url changed (+ dummy api_key required!)
- Warm request, 34-token prompt: prompt eval 869 ms (~39 tok/s), gen 12.5 tok/s
- Later "Hi" request: 26 tokens in 119 ms (~220 tok/s) -> probably prefix caching, verify in step 5
- Estimated gen ceiling: ~40 GB/s / 2 GB = ~20 tok/s; measured ~12-15 tok/s

## Step 3 – streaming

### Ollama server config (from `ollama serve` startup log)
- Ollama version 0.33.2, inference compute: cpu, total 16 GiB, available 3.3 GiB at startup
- OLLAMA_KEEP_ALIVE=5m (default) -> models unload after 5 min idle
- OLLAMA_MAX_LOADED_MODELS=0 (auto), OLLAMA_NUM_PARALLEL=1 (one request at a time per model)
- default context 4096

### Raw Ollama stream (curl -sN .../api/chat, "Count to five.")
- One NDJSON line per token ("1", ",", " " are separate tokens)
- ~78 ms between tokens (created_at timestamps) -> ~12.8 tok/s
- TTFT server-side ~150 ms (load 4 ms + prompt 143 ms), full answer 1.19 s

### Through the proxy with OpenAI client (python test_stream.py)
- TTFT: 263 ms | Total: 7593 ms | 36 prompt / 87 completion tokens | 11.7 tok/s
- Server log for same request: load 3.5 ms, prompt_eval 174.8 ms, eval 7343.7 ms, total 7528.1 ms
- Proxy + HTTP overhead: 7593 - 7528 = ~65 ms (<1%)
- TTFT gap: 263 client vs 178 server (load+prompt) -> ~85 ms = overhead + ~1 token step

### Cold start (first request after restart, python test_client.py)
- load 6343 ms, prompt_eval 656 ms, eval 4114 ms, total 11124 ms
- 57% of the first request was model loading; warm requests: load ~5 ms

### Observations
- Streaming: user sees first text after ~0.26 s instead of waiting ~7.6 s
- Timings can't go in HTTP headers when streaming: headers are sent before timings exist -> logged server-side
- Uvicorn logs "200 OK" BEFORE the "ollama stream:" timing line -> status is committed before generation ends.
  Same problem as output guardrails: you can't take back what's already sent.
- Mid-stream errors can only go into the stream as an SSE error event, not as an HTTP status
- Model answered "why is the sea salty" only half right -> a guardrail checks harm, not correctness
