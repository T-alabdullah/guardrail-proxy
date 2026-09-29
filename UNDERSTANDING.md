# Understanding it

Background for the design decisions. Wherever possible, the evidence is something I measured or saw
on my own machine (2019 MacBook Pro, Intel i9-9880H, 16 GB, CPU only). Numbers: [MEASUREMENTS.md](MEASUREMENTS.md).

## 1. The KV cache, and why generation is limited by memory bandwidth

### What the KV cache is

A language model writes one token at a time. To choose the next token, every layer of the model
compares the current token with **all previous tokens** (that's attention). For that it needs a "key"
and a "value" vector for every previous token, in every layer.

Recomputing those for the whole text at every step would be enormously wasteful. So they're computed
once and **stored**: that's the **KV cache**. Each new token only adds its own keys and values.

**Its size is fixed per token.** For llama3.2:3b:
2 (key + value) × 28 layers × 8 KV heads × 128 dimensions × 2 bytes = **~112 KB per token**,
so ~0.45 GB for the default context of 4096 tokens. I checked this: raising the context from 4096 to
8192 to 16384 tokens grew the model's memory from 2.6 to 3.1 to 4.1 GB (+0.5, +1.0 GB), as predicted.

Three consequences I saw directly:

- **Memory is reserved up front.** Ollama allocates the cache for the full context when it loads the model,
  so `ollama ps` shows 2.6 GB even after a one-word prompt.
- **Prefix caching.** If a new prompt starts exactly like the previous one, the cached keys and values can
  be reused. The same 201-token Llama Guard prompt took 913 ms the first time and **61.5 ms** the second
  (15x faster). Unloading a model throws this cache away.
- **Longer context = slower generation.** Every new token must also read the whole cache. With a
  2690-token prompt, that's ~300 MB extra per token, and generation dropped from 14.4 to 9.7 tok/s.

### Why generation is limited by memory bandwidth, not compute

There are two phases, and they have different bottlenecks:

```
Prompt processing ("prefill")              Generation ("decode")
[t1 t2 t3 ... t500] ──> all weights        [t501] ──> all weights ──> [t502] ──> all weights ──> ...
one read of the weights, many tokens       one full read of the weights PER token
=> limited by arithmetic (compute)         => limited by memory bandwidth
```

**Generation:** to produce one token, the model multiplies one vector by every weight matrix. Every weight
(~2 GB for this model) has to travel from memory to the CPU, and is then used for just one multiply-add.
The CPU finishes the arithmetic long before the next weights arrive. So the speed limit is how fast memory
delivers bytes:

- My laptop's memory bandwidth is roughly 40 GB/s. 40 GB/s ÷ 2 GB per token = **~20 tok/s ceiling**.
- Measured: **12.6-13 tok/s**, about 63% of that ceiling.
- Compute actually used: 13 tok/s × ~6.4 billion operations per token = ~0.08 trillion ops/s,
  only **~10-15% of what the CPU can do**. It's mostly waiting.

**Prompt processing:** all prompt tokens are known in advance, so they're processed together
(in batches of up to 512). Each weight is read once and used for many tokens. Now the arithmetic is the limit:

- Measured: **~80-100 tok/s**, ≈ 0.64 trillion ops/s, roughly the arithmetic ceiling of this CPU.

Same model, same hardware, two different bottlenecks. That's why **time to first token** (mostly
prompt processing, e.g. 327 ms) and **throughput** (generation, ~13 tok/s) are separate numbers. On a GPU,
prompt processing gets dramatically faster (far more compute), while generation is still bound by memory bandwidth.

## 2. What Ollama actually is

Running `ps` while both models were loaded showed three processes:

```
ollama serve                    ~50 MB    <- Ollama itself
llama-server --model ... -c 4096 -np 1 -b 512 -ub 512 --flash-attn auto --context-shift --keep 4
llama-server --model ... (same flags)     <- one per loaded model, each on its own port
```

`llama-server` is the HTTP server from **llama.cpp**, the open-source inference engine (built on the
GGML tensor library). **llama.cpp does the actual work:** it loads the quantized weights (GGUF files,
here 4-bit), tokenizes text, runs the math on the CPU (or GPU), manages the KV cache, and samples tokens.

**Ollama is a manager around it.** It:

- downloads and stores models (`ollama pull`, the blob files in `~/.ollama/models`)
- applies each model's **chat template**, turning a list of messages into the exact text the model expects
  (Llama Guard's template is where the category list lives)
- decides which models are in memory, estimates their size, evicts them, and unloads them after the keep-alive
- starts one `llama-server` process per model and routes requests to it
- offers a simple API (native, plus an OpenAI-compatible one)

**What it hides from you** (all visible in those flags, all set by Ollama's defaults):

| Hidden setting | Why it matters |
|---|---|
| Context length (`-c 4096`) | Decides KV cache memory, and how much conversation the model can see |
| Parallel slots (`-np 1`) | Only **one request at a time** per model; everyone else waits in a queue |
| Batch sizes (`-b/-ub 512`) | How prompt processing is chunked |
| `--context-shift --keep 4` | When the context is full, old tokens are **silently dropped**; long conversations lose their beginning without an error |
| Quantization | The `llama3.2:3b` tag is a 4-bit version; you don't choose the quality/size trade-off unless you pick another tag |
| KV cache precision | 16-bit by default; 8-bit would halve its memory |

Ollama is great for running models locally with no configuration. For a production service, those hidden
defaults decide memory use, concurrency, and quality, so you'd want to control them directly.

## 3. Continuous batching

From section 1: during generation, the CPU mostly waits for weights to arrive from memory. So the
obvious trick is: **while the weights are loaded anyway, use them for several users at once.** Computing
the next token for 8 users costs almost the same memory traffic as for 1, so throughput grows nearly
8x until compute becomes the limit. My measurement says ~85% of compute sits idle during generation,
so there's a lot of room.

**Static batching** groups requests and runs them together until **all** are done:

```
Static batching                    Continuous batching
slot 1: AAAAAAAAAA                 slot 1: AAAAAAAAAA
slot 2: BBB.......  <- idle        slot 2: BBBDDDDDDD  <- D joins the moment B finishes
slot 3: CCCCC.....  <- idle        slot 3: CCCCCEEEEE  <- and E when C finishes
new request D waits for A          no one waits for the longest request
```

Short answers leave slots idle, and new requests wait for the slowest one in the batch.

**Continuous batching** (also called in-flight batching) makes the decision **at every single step**:
finished requests leave, waiting ones join immediately. The batch stays full, so the hardware stays busy
and nobody waits for someone else's long answer. That's why serving engines like vLLM, TGI and TensorRT-LLM
are built around it (llama.cpp supports it too, with more than one slot).

**The limit is memory.** Every active request needs its own KV cache. At 16K context, that's 1.8 GB per
request for this small model, so four users need 7.3 GB of cache alone. How many users fit is decided by
KV cache memory. That's why vLLM's **PagedAttention** allocates the cache in small pages as needed,
instead of reserving the full context per request.

**In this project:** Ollama runs with `-np 1`, so there's no batching: a second user queues. My streaming
guardrail runs Llama Guard *in parallel* only because it's a different model in a different process, and
the two then compete for the same memory bandwidth (generation dropped 33%). A proper serving engine could
batch many users' guard checks together, on separate hardware.

## 4. Where this sits in a real platform

This service is one layer. Around it, a real platform needs:

```
client app ──HTTPS──> API gateway ──────────────> guardrail / policy layer ──> inference servers
                      │ authentication (API keys)   (this project)            (vLLM / llama.cpp on GPUs)
                      │ rate limits and quotas                                 guard model on its own hardware
                      │ usage metering -> billing
                      └ logging, audit, tracing, metrics
```

**Authentication and per-user keys.** Right now the proxy ignores the API key; any string works. A real
platform stores keys hashed, maps each to a user/tenant/project, supports rotation and revocation, and
scopes keys (which models they may use). Keys are never written to logs. The key is also what the
**guardrail policy** hangs on (see DECISIONS.md, section 8).

**Rate limits and quotas.** Per key: requests and tokens per minute, and concurrent requests. Hardware
is shared and KV cache memory is finite, so one heavy user must not starve everyone else.

**Usage metering.** Count prompt and completion tokens per key for every request, from the engine's own
counts, not the client's. Tricky cases this project already has: streams cut by the guardrail or by a
client disconnect (generated tokens still cost compute), and blocked requests (my service reports 0 usage,
but Llama Guard did work, and at 2690 tokens it cost more than the model). The platform has to decide
what to charge for, and record guard compute separately either way.

**Logging.** For every request: key/tenant id, model, token counts, TTFT and total latency, guard verdict,
category, policy applied. **Not** raw prompts by default: they can contain personal data. If prompts are
kept for abuse review, then with a short retention and access controls. Blocked requests should be logged
for audit and for **false-positive review**: that's how you'd find out that "How do I kill a Python
process?" is being blocked. Plus tracing across gateway → guard → model, and dashboards
(p50/p95 TTFT, guard latency, block rate per category).

**Other things around it:** routing requests to model versions, load balancing and autoscaling, TLS,
and abuse detection across requests (a single-request guardrail can't see that one key has tried
50 variations of the same blocked prompt).