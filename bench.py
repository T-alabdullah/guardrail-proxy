"""Benchmark the guardrail proxy: guardrails off vs on, streaming vs non-streaming, warm vs cold.

Usage (proxy on :8000 and Ollama on :11434 must be running, WITHOUT the test switch):
    python bench.py                  # 5 prompts x 1 pass + 2 cold rounds
    python bench.py --runs 3         # more samples
Raw results go to results/bench-<timestamp>.csv, the summary is printed.
"""
import argparse
import csv
import os
import statistics
import time
import uuid
from datetime import datetime

import httpx
from openai import OpenAI

PROXY = "http://localhost:8000/v1"
OLLAMA = "http://localhost:11434"
MODEL = "llama3.2:3b"
GUARD_MODEL = "llama-guard3:1b"
PROMPTS = [
    "Explain how a refrigerator keeps food cold.",
    "Give me three tips for writing a clear email.",
    "What is the difference between weather and climate?",
    "Describe how bees make honey.",
    "Why do we have seasons on Earth?",
]

client = OpenAI(base_url=PROXY, api_key="not-needed")
http = httpx.Client(base_url=PROXY, timeout=600)


# ---------- formatting helpers ----------

def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def pct(before, after):
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before * 100


def f0(x):
    return "-" if x is None else f"{x:.0f}"


def f1(x):
    return "-" if x is None else f"{x:.1f}"


def fp(x):
    return "-" if x is None else f"{x:+.0f}%"


def pick(rows, mode, guard, state):
    return [r for r in rows if r["mode"] == mode and r["guard"] == guard and r["state"] == state]


# ---------- measurements ----------

def nonce(prompt: str) -> str:
    """A unique tag in front of every prompt, so no request reuses the KV cache of an identical earlier prompt."""
    return f"[{uuid.uuid4().hex[:8]}] {prompt}"


def run_stream(prompt: str, guard: bool, max_tokens: int) -> dict:
    """Streaming via the OpenAI client: TTFT and generation speed as the caller sees them."""
    start = time.perf_counter()
    first = last = None
    usage = None
    finish = None
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_tokens,
        temperature=0,
        extra_body={"guardrails": guard},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices:
            c = chunk.choices[0]
            if c.delta.content:
                now = time.perf_counter()
                first = first or now
                last = now
            if c.finish_reason:
                finish = c.finish_reason
    end = time.perf_counter()

    tokens = usage.completion_tokens if usage else None
    gen_tps = None
    if tokens and tokens > 1 and first and last > first:
        gen_tps = (tokens - 1) / (last - first)
    return {
        "ttft_ms": (first - start) * 1000 if first else None,
        "total_ms": (end - start) * 1000,
        "gen_tps": gen_tps,
        "tail_ms": (end - last) * 1000 if last else None,  # after the last token: the guard's final check
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": tokens,
        "finish_reason": finish,
    }


def run_nonstream(prompt: str, guard: bool, max_tokens: int) -> dict:
    """Non-streaming via plain HTTP, so we can read the proxy's timing headers."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "guardrails": guard,
    }
    start = time.perf_counter()
    r = http.post("/chat/completions", json=body)
    total = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    data = r.json()

    def num(key):
        v = r.headers.get(key)
        return float(v) if v else None

    eval_ms = num("x-ollama-eval-ms")
    ollama_total = num("x-ollama-total-ms") or 0.0
    guard_ms = num("x-guard-ms") or 0.0
    tokens = data["usage"]["completion_tokens"]
    return {
        "ttft_ms": total,  # without streaming the caller sees nothing until the very end
        "total_ms": total,
        "load_ms": num("x-ollama-load-ms"),
        "prompt_eval_ms": num("x-ollama-prompt-eval-ms"),
        "eval_ms": eval_ms,
        "ollama_total_ms": ollama_total,
        "guard_ms": guard_ms,
        "other_ms": total - ollama_total - guard_ms,  # proxy + HTTP + everything else
        "gen_tps": tokens / (eval_ms / 1000) if eval_ms else None,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": tokens,
        "finish_reason": data["choices"][0]["finish_reason"],
    }


def unload_all():
    """Ask Ollama to unload both models (keep_alive=0) and wait until memory is free: a cold start."""
    with httpx.Client(base_url=OLLAMA, timeout=60) as h:
        for m in (MODEL, GUARD_MODEL):
            h.post("/api/generate", json={"model": m, "keep_alive": 0})
        for _ in range(40):
            if not h.get("/api/ps").json().get("models"):
                return
            time.sleep(0.5)
    print("  (warning: models still loaded after 20 s)")


# ---------- summary ----------

def summarize(rows):
    print("\n=== Summary (medians) ===")
    print(f"{'mode':10}{'state':6}{'guard':7}{'n':>3}{'TTFT ms':>10}{'total ms':>10}"
          f"{'tok/s':>7}{'guard ms':>10}{'tail ms':>9}")
    for mode in ("stream", "nonstream"):
        for state in ("warm", "cold"):
            for guard in (False, True):
                sel = pick(rows, mode, guard, state)
                if sel:
                    print(f"{mode:10}{state:6}{str(guard):7}{len(sel):>3}"
                          f"{f0(med(sel, 'ttft_ms')):>10}{f0(med(sel, 'total_ms')):>10}"
                          f"{f1(med(sel, 'gen_tps')):>7}{f0(med(sel, 'guard_ms')):>10}"
                          f"{f0(med(sel, 'tail_ms')):>9}")

    s_off, s_on = pick(rows, "stream", False, "warm"), pick(rows, "stream", True, "warm")
    n_off, n_on = pick(rows, "nonstream", False, "warm"), pick(rows, "nonstream", True, "warm")

    print("\n=== What the guardrail adds (warm, medians) ===")
    for label, key in (("TTFT", "ttft_ms"), ("total", "total_ms")):
        a, b = med(s_off, key), med(s_on, key)
        print(f"Streaming {label:7}: {f0(a):>6} -> {f0(b):>6} ms   ({fp(pct(a, b))})")
    a, b = med(s_off, "gen_tps"), med(s_on, "gen_tps")
    print(f"Streaming gen    : {f1(a):>6} -> {f1(b):>6} tok/s ({fp(pct(a, b))})")
    a, b = med(n_off, "total_ms"), med(n_on, "total_ms")
    print(f"Non-stream total : {f0(a):>6} -> {f0(b):>6} ms   ({fp(pct(a, b))})")

    print("\n=== Where the time goes (non-streaming, medians) ===")
    parts = (("model load", "load_ms"), ("prompt processing", "prompt_eval_ms"),
             ("generation", "eval_ms"), ("guard (in + out)", "guard_ms"), ("proxy/HTTP/other", "other_ms"))
    for label, sel in (("warm, guard off", n_off), ("warm, guard on", n_on),
                       ("cold, guard off", pick(rows, "nonstream", False, "cold"))):
        if not sel:
            continue
        total = med(sel, "total_ms")
        print(f"{label}: total {f0(total)} ms")
        for name, key in parts:
            v = med(sel, key) or 0.0
            print(f"   {name:19}{f0(v):>8} ms {v / total * 100:6.1f}%")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=1, help="warm passes over the prompt list")
    ap.add_argument("--cold-runs", type=int, default=2, help="rounds with both models unloaded first")
    ap.add_argument("--max-tokens", type=int, default=100)
    args = ap.parse_args()
    rows = []

    def record(mode, guard, state, prompt, result):
        rows.append({"mode": mode, "guard": guard, "state": state, "prompt": prompt, **result})
        print(f"  {mode:9} guard={str(guard):5} {state}: "
              f"TTFT {f0(result['ttft_ms']):>6} ms, total {f0(result['total_ms']):>6} ms")

    print("Warm-up: loading both models (not recorded)")
    run_nonstream(nonce("Say hi."), True, 5)

    print(f"\nWarm: {args.runs} pass(es) over {len(PROMPTS)} prompts, guard off/on interleaved")
    for _ in range(args.runs):
        for p in PROMPTS:
            for guard in (False, True):
                record("stream", guard, "warm", p, run_stream(nonce(p), guard, args.max_tokens))
            for guard in (False, True):
                record("nonstream", guard, "warm", p, run_nonstream(nonce(p), guard, args.max_tokens))

    print(f"\nCold: {args.cold_runs} round(s), both models unloaded before each request")
    for i in range(args.cold_runs):
        p = PROMPTS[i % len(PROMPTS)]
        for guard in (False, True):
            unload_all()
            record("stream", guard, "cold", p, run_stream(nonce(p), guard, args.max_tokens))
        unload_all()
        record("nonstream", False, "cold", p, run_nonstream(nonce(p), False, args.max_tokens))

    os.makedirs("results", exist_ok=True)
    path = f"results/bench-{datetime.now():%Y%m%d-%H%M%S}.csv"
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRaw results: {path}")
    summarize(rows)


if __name__ == "__main__":
    main()