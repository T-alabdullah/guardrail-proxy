"""Streaming through the proxy with the unmodified OpenAI client, measuring TTFT and throughput."""
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

start = time.perf_counter()
first_token_at = None
usage = None

stream = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Explain in three sentences why the sea is salty."}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.usage:
        usage = chunk.usage
    if chunk.choices and chunk.choices[0].delta.content:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        print(chunk.choices[0].delta.content, end="", flush=True)

end = time.perf_counter()

print("\n")
print(f"TTFT:        {(first_token_at - start) * 1000:.0f} ms")
print(f"Total:       {(end - start) * 1000:.0f} ms")
if usage:
    gen_tps = (usage.completion_tokens - 1) / (end - first_token_at)
    print(f"Tokens:      {usage.prompt_tokens} prompt, {usage.completion_tokens} completion")
    print(f"Generation:  {gen_tps:.1f} tokens/s (after first token)")