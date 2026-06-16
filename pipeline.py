"""
multi-agent-research-pipeline
=============================
Turn one research question into a *verified* brief by orchestrating several LLM
calls in three stages:

    fan-out research  ->  adversarial verification  ->  synthesis

Why bother (vs. one big prompt)?
  * A single pass hallucinates and you can't tell which parts to trust.
  * Splitting the work lets a *different* model instance try to REFUTE each claim,
    so weak claims get filtered before they reach the final answer.
  * Tiering models by job (cheap for breadth, stronger for judgement) keeps it fast
    and inexpensive.

Provider: Anthropic Claude. `complete()` is the only provider-specific function —
swap it to use another LLM.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass

try:
    from anthropic import AsyncAnthropic
except ImportError:
    sys.exit("anthropic SDK not found — run: pip install -r requirements.txt")

# --- model tiers: match the model to the job (breadth = cheap, judgement = strong) ---
MODEL_RESEARCH = os.getenv("MODEL_RESEARCH", "claude-haiku-4-5-20251001")  # broad, cheap
MODEL_VERIFY = os.getenv("MODEL_VERIFY", "claude-sonnet-4-6")              # skeptical judge
MODEL_SYNTH = os.getenv("MODEL_SYNTH", "claude-sonnet-4-6")               # final writer
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "6"))

ANGLES = ["key facts", "risks / caveats", "common misconceptions", "the contrarian view"]

client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment
_sem = asyncio.Semaphore(MAX_CONCURRENCY)


@dataclass
class Claim:
    angle: str
    text: str
    verified: bool = False
    reason: str = ""


async def complete(model: str, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """One concurrency-capped LLM call. The only provider-specific code in the file."""
    async with _sem:
        msg = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are precise and concise.",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()


def _json(text: str):
    """Best-effort extraction of a JSON object/array from a model reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        start, end = text.find("["), text.rfind("]")
    return json.loads(text[start:end + 1])


# ---- stage 1: fan out across angles -> candidate claims ----
async def research(question: str, angle: str) -> list[Claim]:
    out = await complete(
        MODEL_RESEARCH,
        f'Research question: "{question}"\n'
        f'From the angle of "{angle}", list 3 specific, checkable claims. '
        f'Return JSON only: [{{"text": "..."}}, ...].',
    )
    try:
        return [Claim(angle, c["text"]) for c in _json(out)]
    except Exception:
        return []


# ---- stage 2: adversarial verification (a different instance tries to REFUTE) ----
async def verify(question: str, claim: Claim) -> Claim:
    out = await complete(
        MODEL_VERIFY,
        f'Question: "{question}"\nClaim to fact-check: "{claim.text}"\n'
        "Try to REFUTE it. If it is well-supported and accurate, accept it; if it is "
        "wrong, unverifiable, or misleading, reject it. Default to reject when unsure. "
        'Return JSON only: {"verified": true|false, "reason": "..."}.',
        system="You are a skeptical fact-checker. You are rewarded for being right, not agreeable.",
    )
    try:
        verdict = _json(out)
        claim.verified = bool(verdict.get("verified"))
        claim.reason = verdict.get("reason", "")
    except Exception:
        claim.verified, claim.reason = False, "unparseable verdict -> rejected"
    return claim


# ---- stage 3: synthesize from verified claims only ----
async def synthesize(question: str, claims: list[Claim]) -> str:
    kept = [c for c in claims if c.verified]
    if not kept:
        return "No claims survived verification — nothing trustworthy to report."
    bullets = "\n".join(f"- ({c.angle}) {c.text}" for c in kept)
    return await complete(
        MODEL_SYNTH,
        f'Question: "{question}"\n'
        f"Write a tight, balanced brief using ONLY these verified claims:\n{bullets}\n\n"
        "Do not add any facts that aren't listed above.",
        max_tokens=1200,
    )


async def run(question: str) -> str:
    # fan out across angles, concurrently
    batches = await asyncio.gather(*(research(question, a) for a in ANGLES))
    claims = [c for batch in batches for c in batch]

    # adversarially verify every candidate claim, concurrently
    claims = list(await asyncio.gather(*(verify(question, c) for c in claims)))
    survived = sum(c.verified for c in claims)
    print(f"[pipeline] {len(claims)} claims gathered -> {survived} survived verification", file=sys.stderr)

    return await synthesize(question, claims)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python pipeline.py "your research question"')
    print(asyncio.run(run(" ".join(sys.argv[1:]))))
