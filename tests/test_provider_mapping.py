"""Request-mapping tests for each provider adapter (no network)."""

import base64

import pytest

from pai_sdk import (
    AssistantModelMessage,
    FilePart,
    ImagePart,
    JsonOutput,
    TextOutput,
    TextPart,
    ToolCallPart,
    ToolModelMessage,
    ToolResultPart,
    UserModelMessage,
)
from pai_sdk._prompt import standardize_prompt
from pai_sdk.provider import CallOptions, FunctionToolSpec
from pai_sdk.providers import resolve_model_string
from pai_sdk.providers.anthropic import AnthropicLanguageModel
from pai_sdk.providers.google import GoogleLanguageModel, convert_to_gemini_contents
from pai_sdk.providers.openai_chat import OpenAIChatLanguageModel, convert_to_chat_messages
from pai_sdk.providers.openai_responses import (
    OpenAIResponsesLanguageModel,
    convert_to_responses_input,
)
from pai_sdk.errors import NoSuchProviderError

PNG = b"\x89PNG\r\n\x1a\nfake"
PNG_B64 = base64.b64encode(PNG).decode()

CONVO = [
    UserModelMessage(
        content=[
            TextPart(text="What's in this image?"),
            ImagePart(image=PNG),
        ]
    ),
    AssistantModelMessage(
        content=[
            TextPart(text="Checking."),
            ToolCallPart(tool_call_id="call_1", tool_name="lookup", input={"q": "x"}),
        ]
    ),
    ToolModelMessage(
        content=[
            ToolResultPart(
                tool_call_id="call_1",
                tool_name="lookup",
                output=JsonOutput(value={"answer": 42}),
            )
        ]
    ),
]

TOOLS = [
    FunctionToolSpec(
        name="lookup",
        description="look things up",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
]


def options(messages, **kwargs):
    return CallOptions(prompt=messages, **kwargs)


# --- OpenAI Chat Completions -------------------------------------------------


def test_chat_messages_mapping():
    converted = convert_to_chat_messages(
        standardize_prompt(system="be terse", messages=CONVO)
    )
    assert converted[0] == {"role": "system", "content": "be terse"}
    user = converted[1]
    assert user["content"][0] == {"type": "text", "text": "What's in this image?"}
    assert user["content"][1]["type"] == "image_url"
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assistant = converted[2]
    assert assistant["content"] == "Checking."
    assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    tool_msg = converted[3]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"answer": 42}',
    }


def test_chat_request_params():
    model = OpenAIChatLanguageModel(model_id="gpt-5.4")
    request = model._request(
        options(
            standardize_prompt(prompt="hi"),
            max_output_tokens=100,
            temperature=0.5,
            tools=TOOLS,
            tool_choice={"type": "tool", "tool_name": "lookup"},
            provider_options={"openai": {"reasoning_effort": "high"}},
        ),
        stream=True,
    )
    assert request["max_completion_tokens"] == 100
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["function"]["name"] == "lookup"
    assert request["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert request["stream_options"] == {"include_usage": True}
    assert request["extra_body"] == {"reasoning_effort": "high"}


# --- OpenAI Responses --------------------------------------------------------


def test_responses_input_mapping():
    items = convert_to_responses_input(CONVO)
    assert items[0]["content"][1]["type"] == "input_image"
    assert items[1]["content"][0] == {"type": "output_text", "text": "Checking."}
    function_call = items[2]
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == "call_1"
    output = items[3]
    assert output == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"answer": 42}',
    }


def test_responses_request_params():
    model = OpenAIResponsesLanguageModel(model_id="gpt-5.4")
    request = model._request(
        options(
            standardize_prompt(system="sys", prompt="hi"),
            max_output_tokens=50,
            tools=TOOLS,
            tool_choice="required",
            provider_options={"openai": {"reasoning": {"effort": "high"}}},
        )
    )
    assert request["instructions"] == "sys"
    assert request["max_output_tokens"] == 50
    assert request["tools"][0] == {
        "type": "function",
        "name": "lookup",
        "description": "look things up",
        "parameters": TOOLS[0].input_schema,
    }
    assert request["tool_choice"] == "required"
    assert request["extra_body"] == {"reasoning": {"effort": "high"}}


# --- Anthropic ---------------------------------------------------------------


def test_anthropic_request_mapping():
    model = AnthropicLanguageModel(model_id="claude-opus-4-8")
    request = model._request(
        options(
            standardize_prompt(system="sys", messages=CONVO),
            max_output_tokens=1000,
            tools=TOOLS,
            tool_choice="auto",
            provider_options={"anthropic": {"thinking": {"type": "adaptive"}}},
        )
    )
    assert request["system"] == "sys"
    assert request["max_tokens"] == 1000
    user = request["messages"][0]
    assert user["content"][1]["type"] == "image"
    assert user["content"][1]["source"]["type"] == "base64"
    assert user["content"][1]["source"]["media_type"] == "image/png"
    assistant = request["messages"][1]
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {"q": "x"},
    }
    tool_result = request["messages"][2]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "call_1"
    assert request["tools"][0]["input_schema"] == TOOLS[0].input_schema
    assert request["tool_choice"] == {"type": "auto"}
    assert request["extra_body"] == {"thinking": {"type": "adaptive"}}


def test_anthropic_pdf_document():
    model = AnthropicLanguageModel(model_id="claude-opus-4-8")
    request = model._request(
        options(
            [
                UserModelMessage(
                    content=[
                        FilePart(
                            data=b"%PDF-1.4 fake",
                            media_type="application/pdf",
                            filename="doc.pdf",
                        ),
                        TextPart(text="Summarize"),
                    ]
                )
            ]
        )
    )
    block = request["messages"][0]["content"][0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    assert block["title"] == "doc.pdf"


def test_anthropic_thinking_replay():
    from pai_sdk.messages import ReasoningPart

    model = AnthropicLanguageModel(model_id="claude-opus-4-8")
    request = model._request(
        options(
            [
                UserModelMessage(content="hi"),
                AssistantModelMessage(
                    content=[
                        ReasoningPart(
                            text="thinking...",
                            provider_options={"anthropic": {"signature": "sig123"}},
                        ),
                        ReasoningPart(text="unsigned — must be dropped"),
                        TextPart(text="answer"),
                    ]
                ),
                UserModelMessage(content="and?"),
            ]
        )
    )
    assistant = request["messages"][1]
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "thinking...",
        "signature": "sig123",
    }
    assert len(assistant["content"]) == 2  # unsigned reasoning dropped


# --- Google ------------------------------------------------------------------


async def test_gemini_contents_mapping():
    contents = await convert_to_gemini_contents(CONVO)
    user = contents[0]
    assert user["role"] == "user"
    assert user["parts"][0] == {"text": "What's in this image?"}
    assert user["parts"][1]["inline_data"]["mime_type"] == "image/png"
    assert user["parts"][1]["inline_data"]["data"] == PNG
    model_turn = contents[1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][1]["function_call"]["name"] == "lookup"
    tool_turn = contents[2]
    assert tool_turn["role"] == "user"
    assert tool_turn["parts"][0]["function_response"]["response"] == {"answer": 42}


async def test_gemini_config():
    model = GoogleLanguageModel(model_id="gemini-2.5-flash")
    _, config = await model._build(
        options(
            standardize_prompt(system="sys", prompt="hi"),
            max_output_tokens=99,
            tools=TOOLS,
            tool_choice={"type": "tool", "tool_name": "lookup"},
            provider_options={"google": {"thinking_config": {"include_thoughts": True}}},
        )
    )
    assert config["system_instruction"] == "sys"
    assert config["max_output_tokens"] == 99
    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration["name"] == "lookup"
    assert declaration["parameters_json_schema"] == TOOLS[0].input_schema
    assert config["tool_config"]["function_calling_config"] == {
        "mode": "ANY",
        "allowed_function_names": ["lookup"],
    }
    assert config["thinking_config"] == {"include_thoughts": True}


# --- model string resolution -------------------------------------------------


def test_resolve_model_strings():
    assert resolve_model_string("openai/gpt-5.4").provider == "openai.responses"
    assert resolve_model_string("anthropic/claude-opus-4-8").provider == "anthropic"
    assert resolve_model_string("google/gemini-2.5-flash").provider == "google.generative-ai"
    openrouter_model = resolve_model_string("openrouter/google/gemini-2.5-flash")
    assert openrouter_model.provider == "openrouter"
    assert openrouter_model.model_id == "google/gemini-2.5-flash"
    with pytest.raises(NoSuchProviderError):
        resolve_model_string("nope/model")
    with pytest.raises(NoSuchProviderError):
        resolve_model_string("just-a-model")


def test_tool_result_text_output():
    converted = convert_to_chat_messages(
        [
            ToolModelMessage(
                content=[
                    ToolResultPart(
                        tool_call_id="c1",
                        tool_name="t",
                        output=TextOutput(value="plain text"),
                    )
                ]
            )
        ]
    )
    assert converted[0]["content"] == "plain text"
