"""Prompt processing vs generation: how both speeds change with prompt length.

Usage (proxy on :8000 and Ollama on :11434 running):
    python longprompt.py 2>&1 | tee results/longprompt-run1.txt
"""
import csv
import os
import uuid
from datetime import datetime

import httpx

PROXY = "http://localhost:8000/v1"
MODEL = "llama3.2:3b"
SIZES = [5, 25, 75, 150, 220]   # filler lines -> roughly 60 to 2700 prompt tokens (context is 4096)
COLORS = ["red", "blue", "green", "yellow", "white"]
http = httpx.Client(base_url=PROXY, timeout=900)


def build_prompt(n: int) -> str:
    # The random tag at the start makes every prompt unique, so nothing comes from the prefix cache.
    lines = [f"Box {i} on the {COLORS[i % 5]} shelf holds {i * 7 % 97} items." for i in range(1, n + 1)]
    return (f"[{uuid.uuid4().hex[:8]}] Here is an inventory list.\n" + "\n".join(lines)
            + "\n\nHow many boxes are listed? Answer in one short sentence.")


def run(n: int, guard: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(n)}],
        "max_tokens": 30,
        "temperature": 0,
        "guardrails": guard,
    }
    r = http.post("/chat/completions", json=body)
    r.raise_for_status()
    usage = r.json()["usage"]
    prompt_ms = float(r.headers.get("x-ollama-prompt-eval-ms") or 0)
    eval_ms = float(r.headers.get("x-ollama-eval-ms") or 0)
    return {
        "lines": n,
        "guard": guard,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "prompt_ms": prompt_ms,
        "eval_ms": eval_ms,
        "prompt_tps": usage["prompt_tokens"] / (prompt_ms / 1000) if prompt_ms else 0.0,
        "gen_tps": usage["completion_tokens"] / (eval_ms / 1000) if eval_ms else 0.0,
        "guard_ms": float(r.headers.get("x-guard-ms") or 0),
    }


def main():
    rows = []
    print(f"{'lines':>6}{'guard':>7}{'prompt tok':>11}{'prompt ms':>11}"
          f"{'prompt tok/s':>13}{'gen tok/s':>10}{'guard ms':>10}")
    for n in SIZES:
        for guard in (False, True):
            row = run(n, guard)
            rows.append(row)
            print(f"{n:>6}{str(guard):>7}{row['prompt_tokens']:>11}{row['prompt_ms']:>11.0f}"
                  f"{row['prompt_tps']:>13.1f}{row['gen_tps']:>10.1f}{row['guard_ms']:>10.0f}")

    os.makedirs("results", exist_ok=True)
    path = f"results/longprompt-{datetime.now():%Y%m%d-%H%M%S}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRaw results: {path}")


if __name__ == "__main__":
    main()