"""Prompt template sample.

Runs without API keys:

    python examples/prompt_template.py
"""

from pai_sdk import load_prompt


CONFIG = {
    "name": "brace-safe-template",
    "system": (
        'Return JSON shaped like {"status": "ok"} when useful. '
        "You are helping {{company}}."
    ),
    "user": "Ticket: {{ticket}}",
}


def main() -> None:
    prompt = load_prompt(CONFIG)
    messages = prompt.render({"company": "Acme", "ticket": "Login fails"})

    print(f"variables: {', '.join(prompt.variables)}")
    for message in messages:
        print(f"{message.role}: {message.content}")


if __name__ == "__main__":
    main()
