# Structured History And Typed Messages Conversation Summary

Date: 2026-07-06

Status: Working design note

This document summarizes the conversation that led from "should pai-sdk use
GEPA or DSPy?" to the current design direction: pai-sdk should preserve
provider-near `ModelMessage[]` traces while adding typed, hydratable message
definitions as the higher-level developer interface.

It intentionally omits customer names and private trace contents.

## Starting Point

The conversation started with GEPA's `optimize_anything` work. The initial
question was whether pai-sdk should add GEPA, especially the generalized
"optimize any part of text" idea, as an optional dependency.

The target product behavior was:

- accept prompts that currently live in code, JSON Schema, or YAML
- optimize some selected text region
- return an optimized prompt or prompt fragment
- allow optimization of entire system prompts, subsections, skills, tool
  descriptions, or dynamic customer/context sections
- support freezing all other parts of the prompt while optimizing only the
  target region

The early conclusion was that pai-sdk should not embed GEPA itself. GEPA should
be an external optimizer that can use pai-sdk as an inference runner. The
optimizer script should own:

- the dataset
- the optimization loop
- the choice of target text
- the scoring and validation process
- GEPA and LiteLLM dependencies, if it needs them

pai-sdk should own the reliable runtime surface:

- prompt rendering
- model inference
- tool execution
- structured outputs
- replayable traces
- read/apply helpers for explicit optimization targets

## Conversation Trajectory

The conversation was not linear. It moved through several directions that each
felt promising for a while, then exposed a limitation that pushed the design to
the next layer down.

### Phase 1: Maybe GEPA Belongs In Pai-SDK

The first exciting idea was to bring GEPA's generalized optimization ability
into pai-sdk directly. That was attractive because GEPA's `optimize_anything`
language matched the product intuition: users should be able to optimize any
part of a prompt, not only a whole instruction string.

That direction started to look wrong once we separated runtime concerns from
optimization concerns. pai-sdk should be the runner and trace system. GEPA
should own the search loop, datasets, and optimizer dependencies. This avoided
turning pai-sdk into an optimizer framework and kept GEPA/LiteLLM out of the
package.

What survived from this phase:

- optimize arbitrary text targets
- expose stable prompt/message/tool ids
- make optimization scripts choose the target
- keep pai-sdk dependency-light

What we moved against:

- adding GEPA as an optional dependency
- adding `optimize: true` flags to prompt YAML
- making pai-sdk own datasets or optimizer loops

### Phase 2: Maybe Prompt Templates Are Enough

The next attractive idea was simple templating:

```text
You are a helpful assistant.

## Customer context
{{customer_context}}
```

This was compelling because it avoided over-modeling prompt fragments. A prompt
could just declare variables and let an optimizer select either the surrounding
text or one dynamic section.

This remains valid, but it was not enough for the full problem. Templates
render text, but they do not by themselves solve trace replay, structured
history, or hydration from observability logs.

What survived from this phase:

- template variables are a good input-field analog
- optional fields belong in schemas/values
- rendering strategy should handle absent values

What we moved against:

- treating templates alone as the whole abstraction
- assuming reverse hydration is always possible from text alone

### Phase 3: Maybe DSPy Is The Right Runtime

Then the conversation shifted toward DSPy. This was a real contender because
DSPy already has signatures, structured inputs/outputs, adapters, modules,
history, and optimizer familiarity.

The strongest version of this direction was:

- use DSPy signatures and modules
- add a history manager or logging integration
- capture provider-near `ModelMessage[]` alongside DSPy's input/output rows
- convert external agent traces into DSPy history where possible

This felt appealing because it would avoid rebuilding DSPy's structured IO and
module ecosystem. The user explicitly considered abandoning pai-sdk and
building the missing history component into DSPy instead.

That direction weakened once the core product object became clearer. The thing
we need is not only semantic history. It is the exact provider-near conversation
that includes tool calls and tool results in one `ModelMessage[]` array, tied
to semantic inputs and outputs. DSPy's abstractions can maybe be extended to
log this, but it is not the native center of the system.

What survived from this phase:

- DSPy's signatures are a strong ergonomic reference
- DSPy's input/output rows are the right developer-facing semantic shape
- adapters are useful prior art for render/parse lifecycles

What we moved against:

- making DSPy the core runtime dependency
- relying on DSPy's history as the canonical replay object
- assuming MLflow/logging integrations solve provider-near trace replay

### Phase 4: Maybe Pai-SDK Just Needs DSPy-Like Definitions And Tasks

After choosing the pai-native path, the next exciting idea was to add code-first
structured IO:

```python
class ExtractEvent(Definition):
    instructions: str = Instructions("Extract event details from an email.")
    email: str = InputField()
    event_name: str = OutputField()
    date: str = OutputField()
```

Then `Task` could bind that definition to model config and tools:

```python
extract = Task(ExtractEvent, model="openai/gpt-5.4-nano", tools=[...])
```

This was useful and did fill a real gap. It made pai-sdk feel closer to DSPy
for simple structured calls. It also led to a cleaner tool authoring design:
function tools for simple cases, class tools for stateful or explicitly
documented cases.

But this phase also started to feel too narrow. `Task` works when an AI
experience has stable inputs and outputs. Many real agent experiences are not
one stable `input -> output` function. They are message histories with many
typed parts, dynamic sections, imported trace steps, partial known structure,
and raw provider messages that must still be preserved.

What survived from this phase:

- `Definition` is useful as a convenience layer
- `Task` is useful as a module-like bundle
- code-first tool definitions are useful
- typed output models should compile to plain JSON Schema for runtime/replay

What we moved against:

- treating `Task` as the core abstraction
- assuming every experience should fit one stable signature
- over-indexing on DSPy parity instead of message/trace fidelity

### Phase 5: The Stuck / Circling Moment

The final turn was the moment where the design felt like it was looping:

- first optimize prompts
- then use templates
- then maybe use DSPy
- then build pai-native traces
- then add definitions and tasks
- then realize tasks still do not cover the broader message-hydration problem

The user's feeling of being lost was important signal, not confusion to paper
over. It indicated that the abstraction stack had the wrong center. `Task`
answered the DSPy-shaped question, but the actual product need had become more
general:

```python
class Instructions(SystemMessage):
    template: str = """..."""
    variableA: str = Variable()
    variableB: int = Variable()


class Sender(UserMessage):
    template: str = """..."""


class Received(AssistantMessage):
    ...
```

That reframed the whole effort. The center should not be "define a task." The
center should be "define typed messages that hydrate to and from
`ModelMessage[]` when possible."

What survived from the circling:

- pai-sdk should still own provider-near `ModelMessage[]`
- traces still join structured semantics with actual messages
- `Definition` and `Task` are still useful shortcuts

What changed:

- typed/hydratable messages became the likely core abstraction
- `Task` moved down to a convenience or bundle layer
- the next design question became message classes and hydrators, not task
  signatures

## Optimization Targets

We discussed GEPA's "optimize anything" framing, especially the idea that any
text region might be optimizable:

- a whole system prompt
- a subsection of a prompt
- a skill
- a dynamic customer-context section
- a tool description
- a user-message template

The important design correction was that prompt files do not need an
`optimize: true` flag to decide what is optimizable. Optimization intent belongs
to the optimizer script. A config or prompt definition can expose stable ids,
message ids, paths, or named fields, but the optimizer decides which target to
read and rewrite.

The preferred direction became:

- prompt/message definitions expose stable targets
- optimizer scripts select targets explicitly
- pai-sdk provides helpers to read/apply those targets safely
- pai-sdk does not bundle optimizer dependencies

## The Early DSPy Comparison

We mapped DSPy's basic concepts onto pai-sdk:

```text
DSPy instruction   -> pai-sdk system prompt / system messages
DSPy input fields  -> pai-sdk template variables
DSPy output fields -> pai-sdk output schema
```

The initial analogy was useful but incomplete.

DSPy `Signature`s make structured input/output easy. A developer can define
input fields, output fields, and a natural-language instruction. DSPy adapters
then render that signature into a prompt, call a language model, and parse the
result back into structured output.

pai-sdk already had a different center of gravity:

- provider-near `ModelMessage` types inspired by the Vercel AI SDK
- `generate_text` / `stream_text`
- tool-call and tool-result messages
- structured-output helpers
- prompt configs with templates and output schemas
- trace serialization and replay

So the core difference was:

- DSPy owns the signature-to-inference lifecycle.
- pai-sdk should own message structure, traceability, replay, and inference
  helpers while letting providers and the AI SDK style transport stay close to
  the wire.

## Prompt Templates And Optional Fields

We explored whether the system prompt should be modeled as many separately
identified fragments, or simply as a template:

```text
You are a helpful assistant.

## Customer context
{{customer_context}}
```

The preferred mental model was:

- templates are enough for many cases
- variables should be visible in prompt definitions
- optionality belongs in the input schema or structured values
- the hydrator/renderer decides how absent values render

For optional sections, possible rendering strategies are:

- omit the section entirely
- render an empty string
- render an explicit `null` or "not provided"

That decision should belong to the renderer/hydrator, not to the model runtime.

## Hydration

A major thread was the idea of a React-style hydration system for prompts and
messages.

The goal is not only to render structured values to text. It is also to support
the reverse direction when possible:

- structured values render to message text
- rendered text is sent to the model provider
- traces capture the exact rendered `ModelMessage[]`
- later, those traces can be hydrated back into structured objects when the
  hydrator has enough metadata

The reason this matters is observability and replay. If a run is logged to an
observability system, we may later need to reconstruct both:

- the exact model-facing transcript
- the structured application-level inputs/outputs that developers care about

The important limitation is that arbitrary natural-language reverse parsing
cannot be guaranteed. Hydration can be reversible when:

- the template is structured enough
- the original variable metadata is preserved
- the hydrator supports parsing

Otherwise, pai-sdk should fall back to byte-faithful replay from rendered
message content.

## DSPy History Exploration

We then investigated whether this whole effort should move into DSPy instead.

The idea was to create a DSPy "history manager" or logging integration that
stores something richer than DSPy's usual input/output rows:

```python
Span = {
    "id": "...",
    "rootSpanId": "...",
    "module": "...",
    "question": "...",
    "answer": "...",
    "messages": ModelMessage[],
    "usage": {...},
    "metadata": {...},
}

Trace = {
    "id": "...",
    "spans": [...],
}
```

The key requirement was that `messages` is a single ordered array that includes:

- system messages
- user messages
- assistant messages
- tool calls
- local tool results
- final assistant messages

We discussed whether an `lm_calls` array was necessary. The conclusion was that
`lm_calls` would duplicate the important thing if `messages` already contains
the full provider-near transcript. Usage and provider metadata can live as
metadata. The `ModelMessage[]` transcript is the replay object.

## DSPy Normalized LM API

We looked into DSPy's newer normalized LM API changes around DSPy 3.3. The
question was where to intercept real provider messages and associate them with
each DSPy input/output pair.

The explored hypothesis was:

- each `Predict`, `ChainOfThought`, `ReAct`, or other module call creates a
  semantic input/output record
- each such call may involve one or more provider requests
- tool calls and tool responses should be captured as provider-near messages
- a wrapper around the LM or adapter layer might be able to log those messages
  and attach them to the semantic row

This remained theoretically possible, especially through a logging or adapter
integration. However, DSPy's core history model is still primarily semantic:
inputs and outputs, with rendered prompts available via inspection or logging.
It does not naturally make `ModelMessage[]` the canonical history object.

## Bakeoff Framing

We considered a bakeoff between:

1. Extend DSPy with a history manager that captures provider-near messages.
2. Extend pai-sdk with DSPy-like structured input/output and adapters.

The bakeoff goal was not "which library is better." The goal was to test which
path can more naturally produce the target trace object:

```python
Trace = {
    "id": "...",
    "spans": [
        {
            "inputs": {...},
            "outputs": {...},
            "messages": ModelMessage[],
            "usage": {...},
            "metadata": {...},
        }
    ],
}
```

The most important acceptance criteria were:

- provider-near messages are captured in order
- tool calls and tool results are in the same `messages` array
- structured inputs and outputs are attached to the same span
- imported traces from observability systems can recreate the key parts of
  `Trace`
- optimization runners can execute examples and score outputs without owning
  pai-sdk internals

The conclusion from the prototype work was that the pai-native path was more
direct for replay and trace import. DSPy remains useful for inspiration and
possible interop, but it should not be the core runtime dependency.

## Trace Model

The trace model converged on:

```python
Trace = {
    "id": "...",
    "spans": [Span],
}

Span = {
    "id": "...",
    "rootSpanId": "...",
    "parentSpanId": None,
    "inputs": {...},
    "outputs": {...},
    "messages": ModelMessage[],
    "usage": {...},
    "metadata": {...},
}
```

`Trace` is the replayable unit, not just `messages`. The full trace contains
span relationships, structured inputs/outputs, message histories, usage, and
metadata.

Within a span, `messages` is the canonical provider-near transcript. It should
be enough for byte-faithful span replay. If per-step hooks or dynamic runtime
logic modify the effective provider request, that can be stored under metadata
such as `step_request_messages`.

## Observability And Importers

We discussed Braintrust traces and OpenTelemetry/OpenLLMetry-style conversion.
The working view was:

- provider-neutral trace conversion belongs in the package when it is broadly
  useful
- customer-specific Braintrust import examples should stay in examples or
  scripts, not as bundled customer data
- imported traces may have incomplete metadata
- importers should reconstruct the key trace parts where possible

The useful package-level abstraction is probably an OpenTelemetry/OpenLLMetry
converter, because that is general. Braintrust-specific examples can demonstrate
how to map a real observability source into pai-sdk `Trace` without bundling
private trace files.

## Pai-SDK Versus DSPy

The decision shifted away from "pai-sdk versus DSPy" and toward "which
responsibilities belong where."

DSPy strengths:

- signatures
- structured input/output ergonomics
- modules like `Predict`, `ChainOfThought`, and `ReAct`
- optimizers and a broader research ecosystem
- a familiar mental model for many prompt-programming workflows

DSPy gaps relative to the goal:

- no canonical `ModelMessage[]` history alongside each semantic row
- tool calls are not naturally preserved as provider-near tool-call and
  tool-result messages in one transcript
- replay from actual observed provider messages is not the central abstraction
- adapters own much of the render/inference/parse lifecycle

pai-sdk strengths:

- provider-near `ModelMessage[]`
- AI-SDK-style inference API
- tool-call and tool-result messages
- structured outputs
- trace serialization and replay
- prompt configs as data
- optimizer target helpers without optimizer dependencies

pai-sdk gaps relative to DSPy:

- code-first structured input/output needed to be more ergonomic
- prompt/message hydration needed a clearer abstraction
- a module-like `Task` concept was missing
- tools needed a cleaner code-first authoring shape

## Implemented During The Conversation

Several pieces were implemented or drafted during the conversation:

- RFC `0001-trace-backed-structured-history`
- README framing around why pai-sdk exists versus AI SDK and DSPy
- a feature comparison table
- demos before the quickstart showing:
  - simple inference
  - changing the LM
  - adding or swapping fields
  - making the call an agent with tools
  - the history difference between DSPy, AI SDK, and pai-sdk
- `Definition`, `Instructions`, `InputField`, `OutputField`
- `Task` as a runtime wrapper around a definition, model config, tools, and
  execution
- function tool authoring:

```python
def check_calendar(date: str) -> bool:
    return calendar.has_conflict(date)

tools = [
    tool(
        check_calendar,
        description="Check whether a date has scheduling conflicts.",
    )
]
```

- class tool authoring:

```python
class CheckCalendar(Tool):
    description: str = ToolDescription(
        "Check whether a date has scheduling conflicts."
    )
    date: str = ToolInput()
    has_conflict: bool = ToolOutput()

    def forward(self, date: str) -> bool:
        return calendar.has_conflict(date)
```

- support for listing class tools directly when no constructor state is needed:

```python
Task(ExtractEvent, tools=[CheckCalendar])
```

- support for passing initialized tool instances when state is needed:

```python
Task(ExtractEvent, tools=[CheckCalendar(calendar)])
```

## The Latest Shift: Task Is Not The Center

The final turn introduced an important reframing.

`Task` is useful for AI experiences where the inputs and outputs are consistent.
But the broader product goal goes beyond stable `input -> output` tasks.

The more general primitive should be typed/hydratable messages:

```python
class Instructions(SystemMessage):
    template = """
    You are a support agent for {{company}}.

    Policy:
    {{policy}}
    """

    company: str = Variable()
    policy: str | None = Variable(default=None)


class CustomerTicket(UserMessage):
    template = """
    Customer: {{customer_name}}
    Ticket: {{ticket}}
    """

    customer_name: str = Variable()
    ticket: str = Variable()


class TriageResult(AssistantMessage):
    urgency: Literal["low", "medium", "high"]
    summary: str
```

Then the normal flow becomes:

```python
messages = [
    Instructions(company="Acme", policy=policy_text),
    CustomerTicket(customer_name="Jane", ticket=ticket_text),
]

result = await generate(
    messages,
    output=TriageResult,
    tools=[...],
)
```

This is broader than `Task` because:

- each message can have its own structured variables
- different parts of a conversation can be typed independently
- imported traces can hydrate known messages but preserve unknown messages
  as raw `ModelMessage`s
- agent conversations do not need to fit one static task schema
- `ModelMessage[]` remains the replay source of truth

## Current Recommended Architecture

The clearest architecture now looks like:

```text
ModelMessage[]              # provider-near source of truth and replay format
Typed Message classes       # SystemMessage, UserMessage, AssistantMessage
Hydrators                   # render typed objects to text, parse back when possible
Task / RunSpec              # optional bundle: messages + model + tools + output
Definition                  # convenience sugar for simple DSPy-like cases
```

In this framing:

- `ModelMessage[]` remains the runtime and trace primitive.
- typed message classes are the real developer-facing abstraction.
- hydrators own reversible rendering where possible.
- `Task` or `RunSpec` is a useful bundle, closer to a DSPy module.
- `Definition` is a convenience layer for simple single-turn structured I/O.

The core statement of purpose becomes:

> pai-sdk lets developers define typed messages that compile to provider-near
> `ModelMessage[]`, run them through AI-SDK-style inference, and preserve both
> typed semantics and exact trace history.

## Open Questions

### Naming

`Definition` may be too close to DSPy signatures and too narrow for the broader
message-first design. Possible concepts:

- `MessageDefinition`
- `TypedMessage`
- `SystemMessageDefinition`
- `RunSpec`
- `Task`
- `Definition` as shorthand only

### Message Class API

Open questions:

- should `template` be a plain class attribute or a typed field?
- should variables be declared with `Variable()` or reuse Pydantic fields?
- should assistant output messages be Pydantic models?
- how should raw message content and structured values coexist on one object?
- should message classes render directly to `ModelMessage`, or through a
  hydrator registry?

### Hydration Metadata

To support replay and reverse hydration, we need to decide what metadata is
stored on rendered messages:

- message definition id
- template id or version
- variable values
- rendered content
- hydrator id
- parse strategy

### Unknown Imported Messages

Imported traces will include messages that have no known local message class.
The system should preserve these as raw `ModelMessage`s and only hydrate where
metadata or structure makes it safe.

### Relationship Between Task And Message Classes

`Task` could become:

- a named bundle of message definitions, model config, tools, and output
- a convenience wrapper over `generate(...)`
- a replacement for the current `Definition.to_task(...)`
- or simply one optional layer above typed messages

### GEPA And Optimization

The external optimizer contract still needs to be kept clear:

- no GEPA dependency in pai-sdk
- no LiteLLM dependency in pai-sdk
- optimization scripts decide targets
- pai-sdk exposes stable read/apply target helpers
- traces and structured outputs give optimizers what they need to score runs

## Current Takeaway

The conversation moved through three levels:

1. Prompt optimization: how can GEPA optimize pieces of prompt text?
2. Structured history: how do we keep DSPy-like inputs/outputs with real
   provider transcripts?
3. Typed messages: what is the general abstraction that makes both of the above
   natural?

The current answer is:

- do not make GEPA or DSPy core dependencies
- keep `ModelMessage[]` as the provider-near source of truth
- use `Trace` / `Span` to bind semantic rows to provider transcripts
- build typed/hydratable message definitions as the main developer abstraction
- keep `Task` and `Definition` as useful convenience layers, not the center of
  the architecture
