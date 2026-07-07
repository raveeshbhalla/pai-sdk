"""The portable prompt document, end to end.

One JSON document carries everything model-facing — input contract, message
templates, structured output, a tool interface, and a skill — and the same
file runs unchanged in structured-ai-sdk (TypeScript). Code-first conveniences
(Pydantic models, tool(fn)) compile INTO the document; they never replace it.

    python examples/prompt_document.py

Needs ANTHROPIC_API_KEY (or edit the model line).
"""

import asyncio
import json

from pai_sdk import (
    dump_messages_json,
    dump_trace_json,
    generate_text,
    load_prompt,
    span_feedback,
)

DOCUMENT = {
    "name": "support-agent",
    "model": "anthropic/claude-haiku-4-5",
    "params": {"max_output_tokens": 800},
    "input": {"company": "string", "ticket": "string"},
    "output": {"reply": "string", "escalate": "boolean"},
    "system": (
        "You are the support agent for {{company}}. Look up the customer's "
        "plan before answering account questions."
    ),
    "user": "{{ticket}}",
    "skills": {
        "refunds": {
            "description": "Apply when the customer asks for money back.",
            "instructions": (
                "Refunds at {{company}}: plan 'pro' allows refunds up to $500 "
                "without escalation; anything above must set escalate=true."
            ),
        }
    },
    "tools": {
        "lookup_customer": {
            "description": "Look up the customer's plan and monthly spend.",
            "input": {"customer_email": "string"},
            "output": {"plan": "string", "monthly_spend": "number"},
        }
    },
    "tool_choice": "auto",
    "max_steps": 4,
}


async def main() -> None:
    prompt = load_prompt(DOCUMENT)
    print(f"{prompt.name} @ {prompt.content_hash()}  variables={prompt.variables}")

    variables = {
        "company": "Acme",
        "ticket": "jane@corp.example was double charged $700 last month. Refund it.",
    }

    def lookup_customer(input):
        return {"plan": "pro", "monthly_spend": 900.0}

    # One call returns the semantic row AND the provider-near transcript.
    traced = await prompt.generate_trace(
        variables, handlers={"lookup_customer": lookup_customer}
    )
    span = traced.trace.spans[0]
    print("output:", traced.output)
    print("roles: ", [message.role for message in span.messages])
    print("ids:   ", [getattr(m, "id", None) for m in span.messages[:4]])

    # The transcript is replayable data (templates + bindings preserved)...
    _wire = dump_messages_json(span.messages)
    _trace_wire = dump_trace_json(traced.trace)
    # ...and feeds an external optimizer as diagnostic side information.
    print("feedback keys:", sorted(span_feedback(span, include_transcript=False)))

    # Continue the conversation with one typed turn — no full re-render.
    history = [
        *prompt.render(variables),
        *traced.response.messages,
        prompt.render_message("user", {"ticket": "Also, cancel my add-on seat."}),
    ]
    followup = await generate_text(
        model="anthropic/claude-haiku-4-5", messages=history, max_output_tokens=400
    )
    print(
        f"follow-up ({followup.finish_reason}):",
        followup.text[:120].replace("\n", " ") or "(no text)",
    )

    # The document round-trips as plain JSON — this exact file also runs in
    # structured-ai-sdk with the identical content hash.
    reloaded = load_prompt(json.loads(json.dumps(prompt.to_dict())))
    assert reloaded.content_hash() == prompt.content_hash()


if __name__ == "__main__":
    asyncio.run(main())
