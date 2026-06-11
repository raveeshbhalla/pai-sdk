"""Tests for the Bedrock, Vertex, and Azure providers.

Factory / provider-string / model-id assertions, that the right SDK client
class is constructed, that the inherited request mapping is unchanged, and an
Azure end-to-end run through the real openai SDK with a mocked transport.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pai_sdk import generate_text
from pai_sdk._prompt import standardize_prompt
from pai_sdk.provider import CallOptions, FunctionToolSpec
from pai_sdk.providers import (
    azure,
    bedrock,
    create_azure,
    create_bedrock,
    create_vertex,
    resolve_model_string,
    vertex,
)
from pai_sdk.providers.anthropic import AnthropicLanguageModel
from pai_sdk.providers.azure import (
    AzureChatLanguageModel,
    AzureResponsesLanguageModel,
)
from pai_sdk.providers.bedrock import BedrockAnthropicLanguageModel
from pai_sdk.providers.google import GoogleLanguageModel
from pai_sdk.providers.openai_chat import OpenAIChatLanguageModel
from pai_sdk.providers.openai_responses import OpenAIResponsesLanguageModel
from pai_sdk.providers.vertex import (
    VertexAnthropicLanguageModel,
    VertexGoogleLanguageModel,
)

TOOLS = [
    FunctionToolSpec(
        name="lookup",
        description="look things up",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
]


def options(messages, **kwargs):
    return CallOptions(prompt=messages, **kwargs)


# --- Bedrock -----------------------------------------------------------------


def test_bedrock_factory():
    model = bedrock("anthropic.claude-opus-4-8")
    assert isinstance(model, BedrockAnthropicLanguageModel)
    assert isinstance(model, AnthropicLanguageModel)  # reuse via subclass
    assert model.provider == "bedrock.anthropic"
    # ids pass through verbatim — the prefix is NOT added by the factory
    assert model.model_id == "anthropic.claude-opus-4-8"


def test_bedrock_create_passes_credentials():
    factory = create_bedrock(
        aws_region="us-east-1", aws_access_key="ak", aws_secret_key="sk"
    )
    model = factory("anthropic.claude-opus-4-8")
    assert model.aws_region == "us-east-1"
    assert model.aws_access_key == "ak"
    assert model.aws_secret_key == "sk"


def test_bedrock_client_class():
    anthropic_sdk = pytest.importorskip("anthropic")
    if not hasattr(anthropic_sdk, "AsyncAnthropicBedrock"):
        pytest.skip("anthropic[bedrock] extra not installed")
    model = BedrockAnthropicLanguageModel(
        model_id="anthropic.claude-opus-4-8",
        aws_region="us-east-1",
        aws_access_key="ak",
        aws_secret_key="sk",
    )
    client = model._client()
    assert type(client) is anthropic_sdk.AsyncAnthropicBedrock


def test_bedrock_request_mapping_unchanged():
    # Inherited _request produces the identical Anthropic request shape.
    bedrock_model = BedrockAnthropicLanguageModel(model_id="anthropic.claude-opus-4-8")
    plain = AnthropicLanguageModel(model_id="anthropic.claude-opus-4-8")
    opts = options(
        standardize_prompt(system="sys", prompt="hi"),
        max_output_tokens=100,
        tools=TOOLS,
    )
    assert bedrock_model._request(opts) == plain._request(opts)


# --- Vertex (Gemini) ---------------------------------------------------------


def test_vertex_gemini_factory():
    model = vertex("gemini-2.5-flash")
    assert isinstance(model, VertexGoogleLanguageModel)
    assert isinstance(model, GoogleLanguageModel)
    assert model.provider == "google.vertex"
    assert model.model_id == "gemini-2.5-flash"


def test_vertex_create_passes_project_location():
    factory = create_vertex(project="proj", location="us-east5")
    model = factory("gemini-2.5-flash")
    assert model.project == "proj"
    assert model.location == "us-east5"


def test_vertex_gemini_client_class():
    pytest.importorskip("google.genai")
    from google import genai

    model = VertexGoogleLanguageModel(
        model_id="gemini-2.5-flash", project="proj", location="us-central1"
    )
    client = model._client()
    assert type(client) is genai.Client


async def test_vertex_gemini_config_unchanged():
    vertex_model = VertexGoogleLanguageModel(model_id="gemini-2.5-flash")
    plain = GoogleLanguageModel(model_id="gemini-2.5-flash")
    opts = options(
        standardize_prompt(system="sys", prompt="hi"),
        max_output_tokens=99,
        tools=TOOLS,
    )
    _, vertex_config = await vertex_model._build(opts)
    _, plain_config = await plain._build(opts)
    assert vertex_config == plain_config


# --- Vertex (Anthropic) ------------------------------------------------------


def test_vertex_anthropic_factory():
    model = vertex.anthropic("claude-opus-4-8")
    assert isinstance(model, VertexAnthropicLanguageModel)
    assert isinstance(model, AnthropicLanguageModel)
    assert model.provider == "vertex.anthropic"
    assert model.model_id == "claude-opus-4-8"


def test_vertex_anthropic_client_class():
    anthropic_sdk = pytest.importorskip("anthropic")
    if not hasattr(anthropic_sdk, "AsyncAnthropicVertex"):
        pytest.skip("anthropic[vertex] extra not installed")
    model = VertexAnthropicLanguageModel(
        model_id="claude-opus-4-8", project="proj", location="us-east5"
    )
    client = model._client()
    assert type(client) is anthropic_sdk.AsyncAnthropicVertex


def test_vertex_anthropic_request_mapping_unchanged():
    vertex_model = VertexAnthropicLanguageModel(model_id="claude-opus-4-8")
    plain = AnthropicLanguageModel(model_id="claude-opus-4-8")
    opts = options(standardize_prompt(prompt="hi"), max_output_tokens=100, tools=TOOLS)
    assert vertex_model._request(opts) == plain._request(opts)


# --- Azure -------------------------------------------------------------------


def test_azure_factory():
    responses = azure("my-deploy")
    assert isinstance(responses, AzureResponsesLanguageModel)
    assert isinstance(responses, OpenAIResponsesLanguageModel)
    assert responses.provider == "azure.responses"
    assert responses.model_id == "my-deploy"

    chat = azure.chat("my-deploy")
    assert isinstance(chat, AzureChatLanguageModel)
    assert isinstance(chat, OpenAIChatLanguageModel)
    assert chat.provider == "azure.chat"


def test_azure_create_passes_config():
    factory = create_azure(
        api_key="k",
        azure_endpoint="https://x.openai.azure.com",
        api_version="2024-10-21",
    )
    model = factory("my-deploy")
    assert model.api_key == "k"
    assert model.azure_endpoint == "https://x.openai.azure.com"
    assert model.api_version == "2024-10-21"


def test_azure_client_class():
    openai_sdk = pytest.importorskip("openai")
    responses = AzureResponsesLanguageModel(
        model_id="my-deploy",
        api_key="k",
        azure_endpoint="https://x.openai.azure.com",
        api_version="2024-10-21",
    )
    assert type(responses._client()) is openai_sdk.AsyncAzureOpenAI
    chat = AzureChatLanguageModel(
        model_id="my-deploy",
        api_key="k",
        azure_endpoint="https://x.openai.azure.com",
        api_version="2024-10-21",
    )
    assert type(chat._client()) is openai_sdk.AsyncAzureOpenAI


def test_azure_request_mapping_unchanged():
    azure_model = AzureResponsesLanguageModel(model_id="my-deploy")
    plain = OpenAIResponsesLanguageModel(model_id="my-deploy")
    opts = options(
        standardize_prompt(system="sys", prompt="hi"),
        max_output_tokens=50,
        tools=TOOLS,
    )
    assert azure_model._request(opts) == plain._request(opts)


# --- model string resolution -------------------------------------------------


def test_resolve_new_model_strings():
    bedrock_model = resolve_model_string("bedrock/anthropic.claude-opus-4-8")
    assert bedrock_model.provider == "bedrock.anthropic"
    assert bedrock_model.model_id == "anthropic.claude-opus-4-8"

    vertex_model = resolve_model_string("vertex/gemini-2.5-flash")
    assert vertex_model.provider == "google.vertex"
    assert vertex_model.model_id == "gemini-2.5-flash"

    azure_model = resolve_model_string("azure/my-deploy")
    assert azure_model.provider == "azure.responses"
    assert azure_model.model_id == "my-deploy"


# --- Azure end-to-end through the real openai SDK ----------------------------

openai_sdk = pytest.importorskip("openai")


def azure_chat_model(handler) -> AzureChatLanguageModel:
    model = AzureChatLanguageModel(
        model_id="my-deploy",
        api_key="test",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
    )
    model._client_cache = openai_sdk.AsyncAzureOpenAI(
        api_key="test",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return model


AZURE_CHAT_RESPONSE = {
    "id": "chatcmpl-az",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.4",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Paris."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
}


async def test_azure_chat_e2e():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=AZURE_CHAT_RESPONSE)

    result = await generate_text(
        model=azure_chat_model(handler),
        system="be terse",
        prompt="Capital of France?",
    )
    assert result.text == "Paris."
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 10

    # Hit the Azure deployment URL with the api-version query param.
    sent = requests[0]
    assert "/openai/deployments/my-deploy/chat/completions" in str(sent.url)
    assert sent.url.params.get("api-version") == "2024-10-21"
    body = json.loads(sent.content)
    assert body["messages"][0]["role"] == "system"
