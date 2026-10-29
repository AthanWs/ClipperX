from pathlib import Path
import json
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field
from requests import Session
from requests.exceptions import ConnectionError

from clipperx.planner_contracts import DiscoveryBatchResponse
from clipperx.structured_client import (
    OpenAICompatibleStructuredClient,
    StructuredModelConfig,
)
from clipperx.visual_contracts import VisualDirectorResponse


class FakeResponse:
    ok = True
    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"contract_version":1,"batch_id":"b","cursor_before":0,"cursor_after":2,"candidates":[]}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class FakeSession:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.calls = 0
        self.post_args: tuple[str, dict[str, str], int] | None = None

    def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int,
    ) -> FakeResponse:
        self.calls += 1
        self.post_args = (url, headers, timeout)
        self.payload = json
        return FakeResponse()


def test_structured_client_validates_json_schema_response() -> None:
    session = FakeSession()
    client = OpenAICompatibleStructuredClient(
        StructuredModelConfig(
            provider="cliproxy",
            base_url="http://localhost:8317/v1",
            api_key="test",
            model="light-model",
            context_window_tokens=8192,
        ),
        session=session,
    )

    result = client.generate(
        DiscoveryBatchResponse,
        system_prompt="system",
        user_prompt="user",
    )

    assert result.value.cursor_after == 2
    assert result.usage["total_tokens"] == 15
    assert result.usage["estimated_input_tokens"] > 0
    assert session.calls == 1
    assert session.post_args == (
        "http://localhost:8317/v1/chat/completions",
        {"Content-Type": "application/json", "Authorization": "Bearer test"},
        240,
    )
    assert session.payload is not None
    assert session.payload["model"] == "light-model"
    assert session.payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [{"type": "text", "text": "user"}]},
    ]
    assert session.payload["response_format"]["type"] == "json_schema"
    assert "temperature" not in session.payload
    schema = session.payload["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_structured_client_blocks_estimated_context_overflow() -> None:
    session = FakeSession()
    client = OpenAICompatibleStructuredClient(
        StructuredModelConfig(
            provider="cliproxy",
            base_url="http://localhost:8317/v1",
            api_key="test",
            model="light-model",
            context_window_tokens=4096,
            max_output_tokens=1024,
        ),
        session=session,
    )

    with pytest.raises(ValueError, match="excede el contexto"):
        client.generate(
            DiscoveryBatchResponse,
            system_prompt="system",
            user_prompt="contenido " * 20_000,
        )
    assert session.calls == 0


def test_structured_client_does_not_reuse_prior_messages() -> None:
    session = FakeSession()
    client = OpenAICompatibleStructuredClient(
        StructuredModelConfig(
            provider="cliproxy", base_url="http://localhost:8317/v1", api_key="test",
            model="light-model", context_window_tokens=8192,
        ), session=session,
    )
    client.generate(DiscoveryBatchResponse, system_prompt="first system", user_prompt="first user")
    first = session.payload
    assert first is not None
    client.generate(DiscoveryBatchResponse, system_prompt="second system", user_prompt="second user")
    assert session.payload is not None
    assert first["messages"][0]["content"] == "first system"
    assert session.payload["messages"] == [
        {"role": "system", "content": "second system"},
        {"role": "user", "content": [{"type": "text", "text": "second user"}]},
    ]


PLANTED_SECRET = "SECRET_api-key_body_header_https://private.invalid/?token=secret"


class DiagnosticResponse:
    def __init__(self, status: int, code: object = None) -> None:
        self.status_code = status
        self.ok = status < 400
        self.code = code
        self.headers = {"Authorization": PLANTED_SECRET}
        self.url = PLANTED_SECRET
        self.text = PLANTED_SECRET

    def json(self) -> dict[str, Any]:
        if self.ok:
            return {"choices": [{"message": {"content": json_invalid_response()}}]}
        return {"error": {"message": PLANTED_SECRET, "type": PLANTED_SECRET,
                          "code": self.code}, "extra": PLANTED_SECRET}


def json_invalid_response() -> str:
    return '{"unexpected_secret": "' + PLANTED_SECRET + '"}'


class DiagnosticSession:
    def __init__(self, response: DiagnosticResponse, *, transport: bool = False) -> None:
        self.response = response
        self.transport = transport
        self.calls = 0

    def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int,
    ) -> DiagnosticResponse:
        self.calls += 1
        if self.transport:
            raise ConnectionError(PLANTED_SECRET)
        return self.response


def diagnostic_client(session: DiagnosticSession, *, small_context: bool = False
                      ) -> OpenAICompatibleStructuredClient:
    return OpenAICompatibleStructuredClient(StructuredModelConfig(
        provider="cliproxy", base_url="http://localhost:8317/v1", api_key=PLANTED_SECRET,
        model="light-model", context_window_tokens=4096 if small_context else 65536,
        max_output_tokens=1024 if small_context else 4096,
    ), session=session)


@pytest.mark.parametrize(("status", "code", "expected_code"), [
    (429, "insufficient_quota", "insufficient_quota"),
    (429, "rate_limit_exceeded", "rate_limit_exceeded"),
    (429, "context_length_exceeded", "context_length_exceeded"),
    (429, PLANTED_SECRET, None), (429, None, None),
    (503, PLANTED_SECRET, None), (503, {"code": PLANTED_SECRET}, None),
])
def test_provider_http_failure_is_typed_and_redacts_public_exception(
    status: int, code: object, expected_code: str | None,
) -> None:
    from clipperx.structured_client import ProviderHTTPError

    session = DiagnosticSession(DiagnosticResponse(status, code))
    with pytest.raises(RuntimeError) as caught:
        diagnostic_client(session).generate(DiscoveryBatchResponse,
            system_prompt="system", user_prompt="user")
    error = caught.value
    assert isinstance(error, ProviderHTTPError)
    assert error.provider_http_status == status
    assert error.provider_error_code == expected_code
    public = str(error) + repr(error) + repr(vars(error)) + repr(error.args)
    assert PLANTED_SECRET not in public
    assert "private.invalid" not in public
    assert not hasattr(error, "response")
    assert session.calls == 1


def test_input_budget_has_safe_valueerror_subclass_and_prevents_post() -> None:
    from clipperx.structured_client import InputBudgetError

    session = DiagnosticSession(DiagnosticResponse(503, PLANTED_SECRET))
    with pytest.raises(ValueError) as caught:
        diagnostic_client(session, small_context=True).generate(DiscoveryBatchResponse,
            system_prompt=PLANTED_SECRET, user_prompt=(PLANTED_SECRET + " ") * 20_000,
            images=[Path("must-not-read.png")])
    assert isinstance(caught.value, InputBudgetError)
    assert PLANTED_SECRET not in str(caught.value) + repr(vars(caught.value))
    assert session.calls == 0


class ProtocolResponse(DiagnosticResponse):
    def __init__(self, content: object, *, finish: object = "length",
                 usage: object = None, envelope_error: bool = False) -> None:
        super().__init__(200)
        self.content = content
        self.finish = finish
        self.usage = usage
        self.envelope_error = envelope_error

    def json(self) -> dict[str, object]:
        if self.envelope_error:
            raise json.JSONDecodeError(PLANTED_SECRET, PLANTED_SECRET, 0)
        return {"choices": [{"message": {"content": self.content},
                             "finish_reason": self.finish}], "usage": self.usage,
                "extra": PLANTED_SECRET}


@pytest.mark.parametrize(("content", "kind"), [(None, "empty_content"), ("  ", "empty_content"),
    ("{broken " + PLANTED_SECRET, "json_parse"), (json_invalid_response(), "model_validation")])
def test_structured_failure_keeps_only_safe_reported_metadata(content: object, kind: str) -> None:
    from clipperx.structured_client import StructuredResponseError

    usage = {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125,
             "completion_tokens_details": {"reasoning_tokens": 10, "other": PLANTED_SECRET},
             PLANTED_SECRET: PLANTED_SECRET}
    session = DiagnosticSession(ProtocolResponse(content, usage=usage))
    with pytest.raises(ValueError) as caught:
        diagnostic_client(session).generate(DiscoveryBatchResponse,
            system_prompt="system", user_prompt="user")
    error = caught.value
    assert isinstance(error, StructuredResponseError)
    assert error.failure_kind == kind and error.finish_reason == "length"
    assert error.usage == {"prompt_tokens": 100, "completion_tokens": 25,
                           "total_tokens": 125, "reasoning_tokens": 10}
    public = str(error) + repr(error) + repr(vars(error)) + repr(error.args)
    assert PLANTED_SECRET not in public and "private.invalid" not in public
    assert not hasattr(error, "response") and session.calls == 1


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": True, "completion_tokens": "9",
    "total_tokens": -1, "completion_tokens_details": {"reasoning_tokens": 2.5}}, PLANTED_SECRET])
def test_structured_failure_missing_or_invalid_counters_remain_unknown(usage: object) -> None:
    from clipperx.structured_client import StructuredResponseError

    with pytest.raises(StructuredResponseError) as caught:
        diagnostic_client(DiagnosticSession(ProtocolResponse("{", finish=PLANTED_SECRET,
            usage=usage))).generate(DiscoveryBatchResponse, system_prompt="s", user_prompt="u")
    assert caught.value.finish_reason is None
    assert all(value is None for value in caught.value.usage.values())


def test_validation_issue_locations_and_types_cannot_leak_input_or_unknown_keys() -> None:
    from pydantic import BaseModel, ConfigDict, field_validator
    from pydantic_core import PydanticCustomError
    from clipperx.structured_client import StructuredResponseError

    class Child(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[Child]
        label: str

        @field_validator("label")
        @classmethod
        def reject_label(cls, value: str) -> str:
            raise PydanticCustomError(PLANTED_SECRET, PLANTED_SECRET, {"secret": value})

    content = json.dumps({"items": [{"count": PLANTED_SECRET, PLANTED_SECRET: PLANTED_SECRET}],
                          "label": PLANTED_SECRET, PLANTED_SECRET: PLANTED_SECRET})
    with pytest.raises(StructuredResponseError) as caught:
        diagnostic_client(DiagnosticSession(ProtocolResponse(content))).generate(
            Output, system_prompt="s", user_prompt="u")
    error = caught.value
    assert error.failure_kind == "model_validation"
    assert {"type": "int_parsing", "location": ["items", 0, "count"]} in error.issues
    assert len(error.issues) <= 8
    assert PLANTED_SECRET not in json.dumps(vars(error)) + str(error) + repr(error)


def test_response_json_decode_failure_is_safe_envelope_error() -> None:
    from clipperx.structured_client import StructuredResponseError

    with pytest.raises(StructuredResponseError) as caught:
        diagnostic_client(DiagnosticSession(ProtocolResponse(None, envelope_error=True))).generate(
            DiscoveryBatchResponse, system_prompt="s", user_prompt="u")
    assert caught.value.failure_kind == "response_envelope"
    assert caught.value.finish_reason is None
    assert all(value is None for value in caught.value.usage.values())
    assert PLANTED_SECRET not in str(caught.value) + repr(vars(caught.value))


@pytest.mark.parametrize("content", [42, {PLANTED_SECRET: PLANTED_SECRET}, [PLANTED_SECRET]])
def test_unexpected_content_shape_is_envelope_failure(content: object) -> None:
    from clipperx.structured_client import StructuredResponseError

    with pytest.raises(StructuredResponseError) as caught:
        diagnostic_client(DiagnosticSession(ProtocolResponse(content))).generate(
            DiscoveryBatchResponse, system_prompt="s", user_prompt="u")
    assert caught.value.failure_kind == "response_envelope"
    assert PLANTED_SECRET not in str(caught.value) + repr(vars(caught.value))


@pytest.mark.parametrize("finish", ["stop", "length", "tool_calls", "content_filter", "function_call",
    PLANTED_SECRET, {PLANTED_SECRET: PLANTED_SECRET}, 7, None])
def test_finish_reason_is_an_allowlist_not_arbitrary_provider_text(finish: object) -> None:
    from clipperx.structured_client import StructuredResponseError

    with pytest.raises(StructuredResponseError) as caught:
        diagnostic_client(DiagnosticSession(ProtocolResponse("{", finish=finish,
            usage={"completion_tokens": 0}))).generate(DiscoveryBatchResponse,
                system_prompt="s", user_prompt="u")
    expected = finish if type(finish) is str and finish != PLANTED_SECRET else None
    assert caught.value.finish_reason == expected
    assert caught.value.usage == {"prompt_tokens": None, "completion_tokens": 0,
                                  "total_tokens": None, "reasoning_tokens": None}
    assert PLANTED_SECRET not in json.dumps(vars(caught.value))


def test_issue_count_is_bounded_and_locations_follow_schema_structure() -> None:
    from pydantic import BaseModel, ConfigDict
    from clipperx.structured_client import StructuredResponseError

    class Child(BaseModel):
        count: int

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[Child]

    for payload, expected in [({"items": [], "count": PLANTED_SECRET}, 1),
                              ({"items": [{"count": PLANTED_SECRET}] * 20}, 8)]:
        with pytest.raises(StructuredResponseError) as caught:
            diagnostic_client(DiagnosticSession(ProtocolResponse(json.dumps(payload)))).generate(
                Output, system_prompt="s", user_prompt="u")
        assert len(caught.value.issues) == expected
        if expected == 1:
            assert caught.value.issues == [{"type": "extra_forbidden", "location": None}]
        assert PLANTED_SECRET not in json.dumps(vars(caught.value))


class SchemaPayloadSession(DiagnosticSession):
    def __init__(self, content: str) -> None:
        super().__init__(ProtocolResponse(content))
        self.payload: dict[str, Any] = {}

    def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int,
    ) -> DiagnosticResponse:
        self.payload = json
        return super().post(url, headers=headers, json=json, timeout=timeout)


@pytest.fixture
def block_schema_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        pytest.fail("Schema tests must never use a real HTTP session")

    monkeypatch.setattr(Session, "request", reject_network)


def assert_strict_schema_nodes(value: object) -> None:
    if isinstance(value, list):
        for item in value:
            assert_strict_schema_nodes(item)
    elif isinstance(value, dict):
        assert "default" not in value
        if "$ref" in value:
            assert list(value) == ["$ref"]
        if isinstance(value.get("properties"), dict):
            assert value["additionalProperties"] is False
            assert value["required"] == list(value["properties"])
        for item in value.values():
            assert_strict_schema_nodes(item)


def test_visual_director_payload_expands_annotated_refs(
    block_schema_network: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted: list[str] = []

    class CapturingCounter:
        def __init__(self, model: str) -> None:
            pass

        def count(self, text: str) -> int:
            counted.append(text)
            return 1

    monkeypatch.setattr("clipperx.structured_client.TiktokenCounter", CapturingCounter)
    session = SchemaPayloadSession('{"status":"review_required","reasoning":"local test"}')
    source = VisualDirectorResponse.model_json_schema()
    diagnostic_client(session).generate(VisualDirectorResponse, system_prompt="s", user_prompt="u")
    format_ = session.payload["response_format"]
    assert format_["type"] == "json_schema" and format_["json_schema"]["strict"] is True
    schema = format_["json_schema"]["schema"]
    assert json.loads(counted[0].split("\n", 2)[2]) == schema
    for definition, field, target in (
        ("VisualLayer", "destination", "NormalizedRect"),
        ("VisualLayer", "fill", "FillMode"),
        ("VisualShot", "source", "TimeRange"),
        ("VisualShot", "output", "TimeRange"),
    ):
        node = schema["$defs"][definition]["properties"][field]
        assert "$ref" not in node
        assert node["description"] == source["$defs"][definition]["properties"][field]["description"]
        assert node["title"] == source["$defs"][target]["title"]
        if target == "FillMode":
            assert node["enum"] == source["$defs"][target]["enum"]
        else:
            assert node["properties"] == schema["$defs"][target]["properties"]
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])
    assert schema["$defs"]["NormalizedRect"]["properties"]["x"]["minimum"] == 0
    assert schema["$defs"]["NormalizedRect"]["properties"]["width"]["exclusiveMinimum"] == 0
    assert schema["$defs"]["NormalizedRect"]["properties"]["width"]["maximum"] == 1
    assert schema["$defs"]["TimeRange"]["properties"]["end"]["exclusiveMinimum"] == 0
    assert schema["properties"]["status"] == {"$ref": "#/$defs/VisualDirectorStatus"}
    assert_strict_schema_nodes(schema)
    assert VisualDirectorResponse.model_json_schema() == source
    assert session.calls == 1


def test_nested_annotated_refs_preserve_constraints_and_use_site_isolation(
    block_schema_network: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Child(BaseModel):
        """Shared child description."""

        count: int = Field(default=2, ge=1, le=3, description="Bounded count")
        labels: list[str] = Field(min_length=1, max_length=2)

    class Nested(BaseModel):
        left: Child = Field(description="Left child", title="Left title")
        right: Child = Field(description="Right child")
        pure: Child

    class Output(BaseModel):
        nested: list[Nested]

    child = {"count": 2, "labels": ["test"]}
    content = json.dumps({"nested": [{"left": child, "right": child, "pure": child}]})
    source = Output.model_json_schema()
    source_before = json.dumps(source, sort_keys=True)

    def model_schema(cls: type[BaseModel]) -> dict[str, Any]:
        return source

    monkeypatch.setattr(Output, "model_json_schema", classmethod(model_schema))
    session = SchemaPayloadSession(content)
    diagnostic_client(session).generate(Output, system_prompt="s", user_prompt="u")
    schema = session.payload["response_format"]["json_schema"]["schema"]
    properties = schema["$defs"]["Nested"]["properties"]
    for name, description in (("left", "Left child"), ("right", "Right child")):
        node = properties[name]
        assert "$ref" not in node
        assert node["description"] == description
        assert node["properties"] == schema["$defs"]["Child"]["properties"]
        assert node["properties"]["count"]["minimum"] == 1
        assert node["properties"]["count"]["maximum"] == 3
        assert node["properties"]["labels"]["minItems"] == 1
        assert node["properties"]["labels"]["maxItems"] == 2
    assert properties["left"]["title"] == "Left title"
    assert properties["right"]["title"] == "Child"
    assert properties["pure"] == {"$ref": "#/$defs/Child"}
    assert schema["$defs"]["Child"]["description"] == "Shared child description."
    properties["left"]["properties"]["count"]["minimum"] = 99
    assert properties["right"]["properties"]["count"]["minimum"] == 1
    assert schema["$defs"]["Child"]["properties"]["count"]["minimum"] == 1
    assert_strict_schema_nodes(schema)
    assert json.dumps(source, sort_keys=True) == source_before
    assert session.calls == 1


@pytest.mark.parametrize(("node", "definitions"), [
    ({"$ref": None, "description": "bad type"}, {}),
    ({"$ref": "#/$defs/", "description": "empty"}, {}),
    ({"$ref": "#/$defs/Missing", "description": "missing"}, {}),
    ({"$ref": "#/$defs/Bad~2name", "description": "bad escape"}, {}),
    ({"$ref": "https://example.invalid/schema", "description": "external"}, {}),
    ({"$ref": "#/$defs/Scalar", "description": "not an object"}, {"Scalar": 4}),
    ({"$ref": "#/$defs/A", "description": "cycle"}, {"A": {"$ref": "#/$defs/A"}}),
    ({"$ref": "#/$defs/A", "description": "indirect cycle"}, {
        "A": {"type": "object", "properties": {"b": {
            "$ref": "#/$defs/B", "description": "nested"}}},
        "B": {"$ref": "#/$defs/A"},
    }),
    ({"$ref": "#/$defs/A", "minimum": 0}, {"A": {"type": "number", "minimum": 1}}),
    ({"$ref": "#/$defs/A", "description": "mixed", "type": "string"}, {
        "A": {"type": "number"}}),
])
def test_invalid_annotation_ref_expansion_rejects_before_post(
    block_schema_network: None, monkeypatch: pytest.MonkeyPatch,
    node: dict[str, Any], definitions: dict[str, Any],
) -> None:
    class Output(BaseModel):
        count: int

    schema = {"type": "object", "properties": {"count": node}, "$defs": definitions}
    before = json.dumps(schema, sort_keys=True)

    def model_schema(cls: type[BaseModel]) -> dict[str, Any]:
        return schema

    monkeypatch.setattr(Output, "model_json_schema", classmethod(model_schema))
    session = SchemaPayloadSession('{"count":1}')
    with pytest.raises(ValueError, match="schema reference"):
        diagnostic_client(session).generate(Output, system_prompt="s", user_prompt="u")
    assert session.calls == 0
    assert json.dumps(schema, sort_keys=True) == before


def test_annotated_recursive_model_keeps_pure_recursive_refs(block_schema_network: None) -> None:
    class Node(BaseModel):
        children: list["Node"] = Field(default_factory=list)

    class Output(BaseModel):
        node: Node = Field(description="Annotated root")

    session = SchemaPayloadSession('{"node":{"children":[]}}')
    diagnostic_client(session).generate(Output, system_prompt="s", user_prompt="u")
    schema = session.payload["response_format"]["json_schema"]["schema"]
    node = schema["properties"]["node"]
    assert "$ref" not in node and node["description"] == "Annotated root"
    assert node["properties"]["children"]["items"] == {"$ref": "#/$defs/Node"}
    assert_strict_schema_nodes(schema)


def test_schema_positions_preserve_keyword_named_properties_in_client_payload(
    block_schema_network: None,
) -> None:
    class KeywordFields(BaseModel):
        ref_field: str = Field(alias="$ref")
        description: str
        default: str = "fallback"

    class Output(BaseModel):
        annotated: KeywordFields = Field(description="Annotated keyword fields")
        pure: KeywordFields

    values = {"$ref": "literal ref", "description": "literal description", "default": "literal"}
    session = SchemaPayloadSession(json.dumps({"annotated": values, "pure": values}))
    result = diagnostic_client(session).generate(Output, system_prompt="s", user_prompt="u")
    schema = session.payload["response_format"]["json_schema"]["schema"]
    for node in (schema["properties"]["annotated"], schema["$defs"]["KeywordFields"]):
        assert list(node["properties"]) == ["$ref", "description", "default"]
        assert node["required"] == ["$ref", "description", "default"]
        assert node["additionalProperties"] is False
        assert "default" not in node["properties"]["default"]
        assert all(field["type"] == "string" for field in node["properties"].values())
    assert schema["properties"]["pure"] == {"$ref": "#/$defs/KeywordFields"}
    assert result.value.annotated.ref_field == "literal ref"
    assert result.value.annotated.default == "literal"
    assert session.calls == 1


def test_schema_positions_preserve_reference_shaped_examples_in_client_payload(
    block_schema_network: None,
) -> None:
    example = {"$ref": "https://example.invalid/literal", "description": "literal text",
               "default": "literal default", "properties": {"value": {"default": 1}}}

    class Output(BaseModel):
        value: str = Field(examples=[example])

    session = SchemaPayloadSession('{"value":"test"}')
    diagnostic_client(session).generate(Output, system_prompt="s", user_prompt="u")
    schema = session.payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["value"]["examples"] == [example]
    assert session.calls == 1


@pytest.mark.parametrize("keyword", ["examples", "enum", "const", "x-opaque"])
@pytest.mark.parametrize("reference_shaped", [True, False])
def test_schema_positions_copy_opaque_annotations_and_literals_without_rewrites(
    block_schema_network: None, monkeypatch: pytest.MonkeyPatch,
    keyword: str, reference_shaped: bool,
) -> None:
    from clipperx.structured_client import _strict_json_schema

    class Output(BaseModel):
        value: str

    literal: dict[str, Any] = {
        "description": "Opaque literal", "default": "Keep this literal key",
        "properties": {"value": {"type": "string", "default": "opaque nested default"}},
    }
    if reference_shaped:
        literal["$ref"] = "https://example.invalid/literal"
    data = [literal] if keyword in {"examples", "enum"} else literal
    source = Output.model_json_schema()
    source[keyword] = data
    before = json.dumps(source, sort_keys=True)

    def model_schema(cls: type[BaseModel]) -> dict[str, Any]:
        return source

    monkeypatch.setattr(Output, "model_json_schema", classmethod(model_schema))
    schema = _strict_json_schema(Output)
    assert schema[keyword] == data
    assert schema[keyword] is not data
    copied_literal = schema[keyword][0] if isinstance(data, list) else schema[keyword]
    copied_literal["properties"]["value"]["default"] = "mutated copy"
    assert json.dumps(source, sort_keys=True) == before


def test_schema_positions_normalize_nested_schema_values_and_arrays_in_client_payload(
    block_schema_network: None,
) -> None:
    class Child(BaseModel):
        count: int = Field(default=2, ge=1)

    class Output(BaseModel):
        alternatives: list[Annotated[Child, Field(description="Array element")] | str]
        fixed: tuple[Annotated[Child, Field(description="Tuple element")], str]
        keyed: dict[str, Annotated[Child, Field(description="Map value")]]

    child = {"count": 2}
    content = json.dumps({"alternatives": [child, "text"], "fixed": [child, "text"],
                          "keyed": {"default": child}})
    session = SchemaPayloadSession(content)
    diagnostic_client(session).generate(Output, system_prompt="s", user_prompt="u")
    schema = session.payload["response_format"]["json_schema"]["schema"]
    properties = schema["properties"]
    nodes = (
        (properties["alternatives"]["items"]["anyOf"][0], "Array element"),
        (properties["fixed"]["prefixItems"][0], "Tuple element"),
        (properties["keyed"]["additionalProperties"], "Map value"),
    )
    for node, description in nodes:
        assert "$ref" not in node
        assert node["description"] == description
        assert node["additionalProperties"] is False
        assert node["required"] == ["count"]
        assert node["properties"]["count"]["minimum"] == 1
        assert "default" not in node["properties"]["count"]
    assert properties["fixed"]["minItems"] == properties["fixed"]["maxItems"] == 2
    assert session.calls == 1
