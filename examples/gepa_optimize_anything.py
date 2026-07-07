"""Optimize a prompt document with GEPA's optimize_anything — externally.

pai-sdk does not depend on GEPA (or LiteLLM). This script is the pattern for
the OTHER side of that boundary: an optimizer runner that owns the search
loop and datasets, uses pai-sdk as the inference/trace runtime, and persists
the winner as a plain JSON prompt document that any pai-sdk or
structured-ai-sdk app loads.

    pip install gepa   # in the runner's environment, not pai-sdk's
    python examples/gepa_optimize_anything.py

The loop:

1. `read_candidate(prompt, targets)` -> `{address: text}` seed candidate.
2. GEPA proposes evolved candidates with the same keys.
3. `apply_candidate(prompt, candidate)` -> a structurally-safe Prompt
   (variable sets, tool/skill names, and schemas preserved by construction).
4. `generate_trace(...)` runs it; the score plus `span_feedback(span)` (the
   trace as actionable side information) go back to GEPA's reflective
   proposer.
5. `apply_candidate(prompt, result.best_candidate).to_dict()` is the
   optimized document — write it to disk, serve it from a prompt service,
   check it into the app repo.
"""

import asyncio
import json
from pathlib import Path

from pai_sdk import (
    apply_candidate,
    load_prompt,
    read_candidate,
    span_feedback,
)

PROMPT = load_prompt(
    {
        "name": "support-triage",
        "model": "anthropic/claude-haiku-4-5",
        "params": {"max_output_tokens": 400},
        "input": {"company": "string", "ticket": "string"},
        "output": {"urgency": ["low", "medium", "high"], "summary": "string"},
        "system": "You triage support tickets for {{company}}.",
        "user": "Ticket: {{ticket}}",
        "skills": {
            "refunds": {
                "description": "Apply when the customer asks for money back.",
                "instructions": "Treat refund requests for {{company}} as high urgency.",
            }
        },
    }
)

# The optimizer run — not the document — decides what text evolves.
TARGETS = ["message:system", "skill:refunds.instructions"]

TRAIN = [
    {"inputs": {"company": "Acme", "ticket": "Refund me now or I sue."}, "urgency": "high"},
    {"inputs": {"company": "Acme", "ticket": "Where do I change my avatar?"}, "urgency": "low"},
    {"inputs": {"company": "Acme", "ticket": "Checkout 500s for all EU users."}, "urgency": "high"},
    {"inputs": {"company": "Acme", "ticket": "Feature idea: dark mode."}, "urgency": "low"},
]
VAL = [
    {"inputs": {"company": "Acme", "ticket": "I was double charged, fix it."}, "urgency": "high"},
    {"inputs": {"company": "Acme", "ticket": "Docs typo on the pricing page."}, "urgency": "low"},
]


def evaluate(candidate: dict, example: dict, **_kwargs):
    """GEPA evaluator: score one candidate on one example, with trace ASI."""

    async def run():
        evolved = apply_candidate(PROMPT, candidate)
        try:
            traced = await evolved.generate_trace(example["inputs"])
        except Exception as exc:  # failed calls still carry a trace
            trace = getattr(exc, "trace", None)
            feedback = span_feedback(trace.spans[0]) if trace else {"error": str(exc)}
            return 0.0, feedback
        span = traced.trace.spans[0]
        score = 1.0 if traced.output["urgency"] == example["urgency"] else 0.0
        return score, {
            "expected_urgency": example["urgency"],
            **span_feedback(span, include_transcript=score == 0.0),
        }

    return asyncio.run(run())


def main() -> None:
    try:
        from gepa.optimize_anything import GEPAConfig, optimize_anything
    except ImportError:
        raise SystemExit(
            "GEPA is intentionally not a pai-sdk dependency. Install it in "
            "this runner's environment first: pip install gepa"
        )

    result = optimize_anything(
        seed_candidate=read_candidate(PROMPT, TARGETS),
        evaluator=evaluate,
        dataset=TRAIN,   # dataset + valset -> generalization mode
        valset=VAL,
        objective=(
            "Triage support tickets into low/medium/high urgency accurately; "
            "refund requests are high urgency."
        ),
        config=GEPAConfig(),
    )

    optimized = apply_candidate(PROMPT, result.best_candidate)
    out_path = Path("support-triage.optimized.json")
    out_path.write_text(json.dumps(optimized.to_dict(), indent=2, ensure_ascii=False))
    print(f"score={result.best_score} hash={optimized.content_hash()} -> {out_path}")
    # Any app now adopts it with load_prompt("support-triage.optimized.json") —
    # same variables, same ids, same schemas: no call-site changes.


if __name__ == "__main__":
    main()
