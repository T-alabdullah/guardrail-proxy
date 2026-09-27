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