"""Strategy : streaming with background output checks, compared to no guardrail.
To simulate an answer that turns unsafe midway, run the proxy with
GUARD_TEST_FAKE_UNSAFE_WORD=twenty"""
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
PROMPT = "Count from one to forty in words, separated by commas. Nothing else."
TRIGGER = "twenty"


def run(guardrails: bool):
    print(f"\n=== guardrails={guardrails} ===")
    start = time.perf_counter()
    first = None
    text = ""
    finish = None
    info = None
    usage = None

    stream = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"guardrails": guardrails},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        c = chunk.choices[0]
        if c.delta.content:
            if first is None:
                first = time.perf_counter()
            text += c.delta.content
            print(c.delta.content, end="", flush=True)
        if c.finish_reason:
            finish = c.finish_reason
            info = getattr(chunk, "guardrail", None)
    end = time.perf_counter()

    print("\n")
    print(f"finish_reason: {finish}")
    print(f"guardrail:     {info}")
    print(f"TTFT:          {(first - start) * 1000:.0f} ms")
    print(f"Total:         {(end - start) * 1000:.0f} ms")
    if usage and usage.completion_tokens > 1:
        print(f"Generation:    {(usage.completion_tokens - 1) / (end - first):.1f} tokens/s "
              f"({usage.completion_tokens} tokens)")
    if TRIGGER in text.lower():
        leaked = text[text.lower().index(TRIGGER):]
        print(f"Shown from first '{TRIGGER}' onward: {len(leaked)} chars")


run(False)
run(True)