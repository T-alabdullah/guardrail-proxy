"""Llama Guard client: asks the guard model whether the last turn of a conversation is safe."""
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

GUARD_MODEL = os.getenv("GUARD_MODEL", "llama-guard3:1b")
log = logging.getLogger("uvicorn.error")

# Llama Guard 3 hazard categories (MLCommons taxonomy). S14 only exists in the larger models.
CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


class GuardError(Exception):
    """The guard model could not give a usable verdict."""


@dataclass
class Verdict:
    safe: bool
    categories: list[str] = field(default_factory=list)
    raw: str = ""
    elapsed_ms: float = 0.0

    def describe(self) -> str:
        return ", ".join(f"{c}: {CATEGORIES.get(c, 'Unknown category')}" for c in self.categories)

    def summary(self) -> str:
        return "safe" if self.safe else f"unsafe {','.join(self.categories)}"


def parse_verdict(text: str) -> tuple[bool, list[str]]:
    """Llama Guard answers 'safe', or 'unsafe' followed by a line like 'S1,S10'."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise GuardError("guard model returned an empty answer")
    first = lines[0].lower()
    if first == "safe":
        return True, []
    if first == "unsafe":
        categories = [c.strip().upper() for c in lines[1].split(",") if c.strip()] if len(lines) > 1 else []
        return False, categories
    raise GuardError(f"unexpected guard answer: {text!r}")


async def classify(client: httpx.AsyncClient, messages: list[dict]) -> Verdict:
    """messages: already normalized (role + string content).
    Only user/assistant turns go to Llama Guard. It judges the LAST turn:
    a user turn last = input check, an assistant turn last = output check."""
    convo = [m for m in messages if m["role"] in ("user", "assistant")]
    if not convo:
        return Verdict(safe=True, raw="(nothing to check)")

    start = time.perf_counter()
    try:
        r = await client.post("/api/chat", json={
            "model": GUARD_MODEL,
            "messages": convo,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 20},
        })
    except httpx.RequestError as e:
        raise GuardError(f"guard model unreachable: {e}") from e
    if r.status_code != 200:
        raise GuardError(f"guard model returned HTTP {r.status_code}: {r.text[:200]}")

    data = r.json()
    raw = data.get("message", {}).get("content", "")
    safe, categories = parse_verdict(raw)
    elapsed_ms = (time.perf_counter() - start) * 1000

    ms = lambda k: data.get(f"{k}_duration", 0) / 1e6
    log.info("guard timings: load=%.1fms prompt_eval=%.1fms (%d tok) eval=%.1fms (%d tok) wall=%.1fms",
             ms("load"), ms("prompt_eval"), data.get("prompt_eval_count", 0),
             ms("eval"), data.get("eval_count", 0), elapsed_ms)
    return Verdict(safe=safe, categories=categories, raw=raw, elapsed_ms=elapsed_ms)