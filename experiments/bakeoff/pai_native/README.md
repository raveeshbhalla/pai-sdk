# Pai-native Trace/Span bakeoff prototype

Worker A prototype for a canonical Pai Trace/Span object:

```python
Trace = { "id": str, "spans": [Span] }
Span = {
  "id": str,
  "rootSpanId": str,
  "parentSpanId": str | None,
  "inputs": dict,
  "outputs": dict,
  "messages": ModelMessage[],
  "usage": dict | None,
  "metadata": dict,
}
```

The script uses an embedded generic content-review row. It converts content
fields into redacted length/count placeholders, then runs a local fixture model
through Pai's real
`generate_text`, `Output.object`, typed prompt rendering, tool loop, and
`dump_messages/load_messages` serializer.

Run:

```bash
PYTHONPATH=src .venv/bin/python experiments/bakeoff/pai_native/trace_prototype.py --pretty
```

Useful checks:

```bash
PYTHONPATH=src .venv/bin/python experiments/bakeoff/pai_native/trace_prototype.py --pretty >/tmp/pai_trace.json
PYTHONPATH=src .venv/bin/python -m pytest tests/test_generate.py tests/test_serialize.py
```

What it demonstrates:

- `inputs`: structured review-row shape, prompt variables, prompt metadata.
- `outputs`: parsed `JudgeOutput`, raw JSON text, finish reason, tool calls/results.
- `messages`: rendered typed system/user messages plus generated assistant/tool
  messages, round-tripped via Pai's public message serializer.
- `usage`: aggregated token usage from the two fake provider steps.
- `metadata`: provider/model ids, fixture source, step finish reasons, response roles.

Current gap: Trace/Span is a thin wrapper in the experiment, not a first-class
SDK result type. That is enough to evaluate shape and replay trustworthiness,
but production GEPA/replay would want an official helper around `GenerateTextResult`.
