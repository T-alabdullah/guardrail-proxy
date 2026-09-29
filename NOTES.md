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

## Llama Guard categories

S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.


## Step 4a – input + output guard (python test_guard.py)
- All tests pass: safe passes, unsafe -> content_filter + S2, stream refusal is a stream, "yes" -> 400
- Llama Guard prompt ~200 tokens for a 7-word user prompt (template + category list); answer 2-5 tokens
- Guard cost = mostly prompt processing: ~1-2.5 s per check on CPU (warm)
- Prefix cache: identical 201-token guard prompt 913 ms -> 61.5 ms second time (15x)
  -> benchmarks must vary prompts or they measure the cache
- Bread (cold): guard in 4840 ms (incl. 2937 load) + model 11637 ms (incl. 6012 load) + guard out 2648 ms
- Blocked request: answered in 282 ms, model never runs (vs 1970 ms unguarded)
- Without guard, llama3.2 refuses hotwiring itself, but finish_reason "stop" -> not machine-readable.
  Guard gives content_filter + S2, and works independent of which model is behind it
- Non-streaming output check: user waits for full generation + a guard check before seeing anything


## Step 4c – streaming + output guard, strategy D (python test_stream_guard.py)
Decision: D = stream immediately, check the growing answer in the background, cut on unsafe.

Run 1 (all safe), "count one to forty", 99 tokens:
- No guard:   TTFT 177 ms, total 8505 ms, 11.8 tok/s
- Guard (D):  TTFT 5044 ms, total 19242 ms, 6.9 tok/s  -> total +126%, gen -42%
- TTFT: input check 4889 ms, of which 2871 ms = Llama Guard reload (unloaded after 5 min keep-alive)
- Generation slower because guard runs on the SAME CPU: ollama eval 8349 -> 13047 ms for 99 tokens
- Checks ran during generation (102, 202, 307 chars) + tail check at 400 chars after generation
- Each check 1.1-2.3 s; guard busy almost continuously; tail check adds ~1.2 s to the end
- Each check re-reads the whole answer; prompt eval stays 1-2 s -> cost grows with answer length

Run 2 (GUARD_TEST_FAKE_UNSAFE_WORD=twenty, simulated unsafe output):
- Guarded stream cut at "twenty-six": finish_reason content_filter, S1, 57 of 99 tokens generated
- "twenty" first sent at ~char 144, cut at char 226 -> ~82 chars (~20 tokens, ~2.5 s) of "unsafe" text leaked
- Log: check 1 at 102 chars safe (1443 ms, "twenty" appeared during it);
  next check only after 100 more chars (STREAM_GUARD_MIN_CHARS); check 2 at 202 chars unsafe (1427 ms)
- Leak window ~= wait until next check starts + check duration
  -> smaller MIN_CHARS = smaller window but more checks + more CPU contention
- TTFT 261 ms: input guard only 142 ms (guard warm + identical prompt in prefix cache) vs 5044 ms cold in run 1

Conclusion D: fast TTFT, but can't prevent leaks; ~20 tokens here. OK for low-severity,
not for S4/S9. Client must honor content_filter. Hybrid (D normally, C for high risk) is the realistic design.


## Step 5a – benchmark (python bench.py 2>&1 | tee results/bench-run1.txt)
Raw: results/bench-20260929-093154.csv. 5 prompts, nonce per request (no prefix-cache reuse),
temperature 0, max_tokens 100, medians. Runs very consistent (±4%).

Guardrail overhead (warm):
- Stream TTFT 327 -> 1306 ms (+300%, +1.0 s = input check)
- Stream total 8159 -> 13913 ms (+71%); gen 12.6 -> 8.5 tok/s (-33%, guard competes for CPU); tail check 976 ms
- Non-stream total 7995 -> 10442 ms (+31%); guard 2371 ms; gen unchanged 13.0 tok/s (sequential, no contention)
- D vs non-stream: first text 8x sooner (1.3 vs 10.4 s) but finishes 3.5 s later on shared hardware

Where the time goes (non-stream):
- Warm, no guard: generation 96%, prompt 3.6%, load 0.1%, proxy 5 ms (0.1%)
- Warm, guard: generation 74%, guard 23%
- Cold: load 3859 ms = 31% (vs 6343 ms at first-ever start: macOS file cache keeps weights in RAM)
- Cold + guard, stream: TTFT 4380 -> 8353 ms (two models to load)

TTFT vs throughput:
- TTFT ~ prompt processing (292 ms / ~40 tok, all tokens at once, ~140 tok/s)
- Throughput ~ generation (12.6-13 tok/s, one token at a time, weights read per token)
- ~63% of estimated 20 tok/s bandwidth ceiling


## Step 5b – keep-alive and memory
Setting: OLLAMA_KEEP_ALIVE (server-wide, default 5m; -1 = never unload),
or "keep_alive" per request (overrides; 0 = unload now).

Ollama = manager: `ps` shows `ollama serve` (50 MB) + one llama.cpp `llama-server`
process per loaded model, each on its own port. Flags Ollama sets and hides:
-c 4096 (context/KV size), -np 1 (one request at a time -> no batching),
-b/-ub 512 (prompt batch), --flash-attn auto, --context-shift --keep 4.
Server binary is from /Applications/Ollama.app (explains the port conflict at setup).

Memory of keeping both loaded:
- ollama ps SIZE: chat 2.6 GB + guard 1.8 GB = 4.4 GB (28% of 16 GB)
- macOS RSS: ~1.5-1.6 GB per llama-server (~3.0 GB): KV cache reserved but mostly untouched
- Plan with SIZE (full context fills the cache)

KV cache experiment (llama3.2:3b, num_ctx via API):
- 4096: 2.6 GB | 8192: 3.1 GB (+0.5) | 16384: 4.1 GB (+1.0)
- Predicted 2 x 28 layers x 8 kv heads x 128 dims x 2 bytes = ~112 KB/token -> +0.45 / +0.9 GB. Matches.
- Weights + buffers ~2.15 GB; KV cache is per parallel slot -> memory limits concurrency

Cost of unloading (from 5a): cold TTFT +4.1 s without guard, +7 s with guard (two models load)

Experiment: OLLAMA_MAX_LOADED_MODELS=1 (only one model in memory), guarded non-stream, max_tokens 50:
- Request 1: 16.6 s, request 2: 13.6 s
- Request 2: guard in load 4 ms (still loaded) -> chat load 3710 ms (evicts guard)
  -> guard out load 2244 ms (evicts chat). ~6 s loading = 44% of total, EVERY request. ~+80% vs no swapping
- Unloading also discards the prefix/KV cache: guard prompt_eval 1501 ms for 201 tokens

Decision: OLLAMA_KEEP_ALIVE=-1 (both models stay loaded).
Cost 4.4 GB (28% of RAM) permanently; saves 4-7 s cold TTFT + keeps prefix cache.
Right when traffic is regular and both models are on the critical path; wrong for rarely used services.