"""Stateless structured-output client for OpenAI-compatible providers."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, get_args

import requests
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorType

from .context_budget import TiktokenCounter

ModelT = TypeVar("ModelT", bound=BaseModel)

_SAFE_PROVIDER_ERROR_CODES = frozenset({
    "insufficient_quota", "rate_limit_exceeded", "context_length_exceeded",
})


class ProviderHTTPError(RuntimeError):
    """Provider rejection with only safe public diagnostics, never response text."""

    def __init__(self, status: int, code: object = None) -> None:
        if type(status) is not int or not 100 <= status <= 599:
            raise ValueError("Provider HTTP status must be an integer from 100 to 599")
        self.provider_http_status = status
        self.provider_error_code = (
            code if type(code) is str and code in _SAFE_PROVIDER_ERROR_CODES else None
        )
        detail = f" ({self.provider_error_code})" if self.provider_error_code else ""
        super().__init__(f"Provider HTTP {status}{detail}")


class InputBudgetError(ValueError):
    """Estimated request exceeds the configured context before any POST."""


_SAFE_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_SAFE_RESPONSE_FAILURES = frozenset({
    "empty_content", "json_parse", "model_validation", "response_envelope",
})
_SAFE_ISSUE_TYPES = frozenset(get_args(ErrorType))


class StructuredResponseError(ValueError):
    """A failed response with only allowlisted diagnostics in public attributes.

    No error message, input, response object or raw content is retained. Usage
    stays unknown unless a nonnegative integer was actually reported. Validation
    issues are capped at eight, with at most sixteen schema-verified path parts.
    """

    def __init__(self, failure_kind: str, *, finish_reason: object = None,
                 usage: object = None, validation_error: ValidationError | None = None,
                 output_model: type[BaseModel] | None = None) -> None:
        self.failure_kind = (failure_kind if failure_kind in _SAFE_RESPONSE_FAILURES
                             else "response_envelope")
        self.finish_reason = (finish_reason if type(finish_reason) is str
                              and finish_reason in _SAFE_FINISH_REASONS else None)
        reported = usage if isinstance(usage, dict) else {}
        self.usage = {key: _reported_counter(reported.get(key)) for key in (
            "prompt_tokens", "completion_tokens", "total_tokens")}
        details = reported.get("completion_tokens_details")
        self.usage["reasoning_tokens"] = _reported_counter(
            details.get("reasoning_tokens") if isinstance(details, dict) else None)
        self.issues = (_safe_validation_issues(validation_error, output_model)
                       if validation_error is not None and output_model is not None else [])
        super().__init__(f"Structured response failed: {self.failure_kind}")


def _reported_counter(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _schema_branches(node: object, schema: dict[str, object], *, depth: int = 0
                     ) -> list[dict[str, object]]:
    """Expand only trusted model schema references/unions, with bounded recursion."""
    if not isinstance(node, dict) or depth >= 16:
        return []
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        definitions = schema.get("$defs")
        target = definitions.get(ref[len("#/$defs/"):]) if isinstance(definitions, dict) else None
        return _schema_branches(target, schema, depth=depth + 1)
    branches = [node]
    for key in ("anyOf", "allOf", "oneOf"):
        alternatives = node.get(key)
        if isinstance(alternatives, list):
            for alternative in alternatives:
                branches.extend(_schema_branches(alternative, schema, depth=depth + 1))
    return branches


def _safe_issue_location(location: tuple[int | str, ...], schema: dict[str, object]
                         ) -> list[int | str] | None:
    if len(location) > 16:
        return None
    nodes: list[dict[str, object]] = [schema]
    for part in location:
        candidates: list[dict[str, object]] = []
        for node in nodes:
            for branch in _schema_branches(node, schema):
                if type(part) is str:
                    properties = branch.get("properties")
                    if isinstance(properties, dict) and part in properties:
                        candidates.extend(_schema_branches(properties[part], schema))
                elif type(part) is int and 0 <= part <= 1_000_000:
                    candidates.extend(_schema_branches(branch.get("items"), schema))
        if not candidates:
            # Drop the whole location instead of fabricating a shorter root path.
            return None
        nodes = candidates
    return list(location)


def _safe_validation_issues(error: ValidationError, output_model: type[BaseModel]
                            ) -> list[dict[str, object]]:
    schema = output_model.model_json_schema()
    issues: list[dict[str, object]] = []
    for detail in error.errors(include_input=False, include_context=False, include_url=False)[:8]:
        issue_type = detail["type"]
        issues.append({"type": issue_type if issue_type in _SAFE_ISSUE_TYPES else None,
                       "location": _safe_issue_location(detail["loc"], schema)})
    return issues


@dataclass(frozen=True, slots=True)
class StructuredModelConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    context_window_tokens: int
    max_output_tokens: int = 4096
    timeout_seconds: int = 240
    supports_json_schema: bool = True
    supports_temperature: bool = False
    input_utilization: float = 0.9
    estimated_tokens_per_image: int = 1024

    def validate(self) -> None:
        if self.provider not in {"cliproxy", "openai_compatible", "ollama"}:
            raise ValueError(f"Provider no soportado por este cliente: {self.provider}")
        if not self.model.strip():
            raise ValueError("Falta seleccionar model")
        if self.context_window_tokens < 4096:
            raise ValueError("context_window_tokens debe ser al menos 4096")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("max_output_tokens debe ser menor que el contexto")
        if not 0.5 <= self.input_utilization <= 0.95:
            raise ValueError("input_utilization debe estar entre 0.5 y 0.95")
        if self.estimated_tokens_per_image < 256:
            raise ValueError("estimated_tokens_per_image debe ser al menos 256")


@dataclass(frozen=True, slots=True)
class StructuredResult(Generic[ModelT]):
    value: ModelT
    usage: dict[str, int]
    raw_response: dict[str, Any]


class _HTTPResponse(Protocol):
    status_code: int

    @property
    def ok(self) -> bool: ...

    def json(self) -> Any: ...


class _HTTPTransport(Protocol):
    def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int,
    ) -> _HTTPResponse: ...


class OpenAICompatibleStructuredClient:
    """No server-side session is assumed; every request contains all required state."""

    def __init__(
        self,
        config: StructuredModelConfig,
        *,
        session: _HTTPTransport | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.session = session or requests.Session()

    def generate(
        self,
        output_model: type[ModelT],
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[Path] | None = None,
        temperature: float = 0.2,
    ) -> StructuredResult[ModelT]:
        image_paths = images or []
        estimated_input_tokens = self._validate_context(
            output_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_count=len(image_paths),
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image)},
                }
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.config.max_output_tokens,
            "response_format": self._response_format(output_model),
        }
        if self.config.supports_temperature:
            payload["temperature"] = temperature
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = self.session.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        if not response.ok:
            raise _provider_error(response)
        try:
            raw = response.json()
        except ValueError:
            # HTTP-body decoding is not evidence of a valid model response.
            raise StructuredResponseError("response_envelope") from None
        usage_raw = raw.get("usage") if isinstance(raw, dict) else None
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise StructuredResponseError("response_envelope", usage=usage_raw)
        choice = choices[0]
        finish = choice.get("finish_reason")
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise StructuredResponseError("response_envelope", finish_reason=finish, usage=usage_raw)
        content_text = message["content"]
        if isinstance(content_text, list):
            if any(not isinstance(item, dict) or not isinstance(item.get("text"), str)
                   for item in content_text):
                raise StructuredResponseError("response_envelope", finish_reason=finish, usage=usage_raw)
            content_text = "".join(item["text"] for item in content_text)
        if content_text is None or isinstance(content_text, str) and not content_text.strip():
            raise StructuredResponseError("empty_content", finish_reason=finish, usage=usage_raw)
        if not isinstance(content_text, str):
            raise StructuredResponseError("response_envelope", finish_reason=finish, usage=usage_raw)
        try:
            cleaned = _strip_code_fence(content_text)
        except json.JSONDecodeError:
            raise StructuredResponseError("json_parse", finish_reason=finish, usage=usage_raw) from None
        try:
            value = output_model.model_validate_json(cleaned)
        except ValidationError as error:
            raise StructuredResponseError("model_validation", finish_reason=finish, usage=usage_raw,
                validation_error=error, output_model=output_model) from None
        usage_raw = raw.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
            "estimated_input_tokens": estimated_input_tokens,
        }
        return StructuredResult(value=value, usage=usage, raw_response=raw)

    def _validate_context(
        self,
        output_model: type[BaseModel],
        *,
        system_prompt: str,
        user_prompt: str,
        image_count: int,
    ) -> int:
        counter = TiktokenCounter(self.config.model)
        schema = json.dumps(_strict_json_schema(output_model), ensure_ascii=False)
        text_tokens = counter.count(f"{system_prompt}\n{user_prompt}\n{schema}")
        estimate = text_tokens + image_count * self.config.estimated_tokens_per_image
        input_limit = (
            int(self.config.context_window_tokens * self.config.input_utilization)
            - self.config.max_output_tokens
        )
        if estimate > input_limit:
            raise InputBudgetError(
                "La solicitud estimada excede el contexto disponible: "
                f"input={estimate}, limite={input_limit}, imagenes={image_count}, "
                f"max_output={self.config.max_output_tokens}"
            )
        return estimate

    def _response_format(self, output_model: type[BaseModel]) -> dict[str, Any]:
        if not self.config.supports_json_schema:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "strict": True,
                "schema": _strict_json_schema(output_model),
            },
        }


def _image_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _strip_code_fence(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    json.loads(cleaned)
    return cleaned


def _provider_error(response: _HTTPResponse) -> ProviderHTTPError:
    try:
        payload = response.json()
    except ValueError:
        return ProviderHTTPError(response.status_code)
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return ProviderHTTPError(response.status_code)
    return ProviderHTTPError(response.status_code, error.get("code"))


def _strict_json_schema(output_model: type[BaseModel]) -> dict[str, Any]:
    schema = output_model.model_json_schema()
    annotations = {"title", "description", "examples", "deprecated", "readOnly", "writeOnly",
                   "$comment"}
    schema_maps = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
    schema_values = {"items", "additionalItems", "additionalProperties", "propertyNames",
                     "contains", "not", "if", "then", "else", "unevaluatedProperties",
                     "unevaluatedItems", "contentSchema"}
    schema_arrays = {"anyOf", "allOf", "oneOf", "prefixItems"}

    def resolve(ref: object) -> dict[str, Any]:
        # Pydantic's local definition pointers only; never fetch external schemas.
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise ValueError("Invalid local schema reference for annotation expansion")
        name = ref[len("#/$defs/"):]
        if not name or "/" in name or "~" in name.replace("~0", "").replace("~1", ""):
            raise ValueError("Malformed local schema reference for annotation expansion")
        name = name.replace("~1", "/").replace("~0", "~")
        definitions = schema.get("$defs")
        target = definitions.get(name) if isinstance(definitions, dict) else None
        if not isinstance(target, dict):
            raise ValueError("Unresolved local schema reference for annotation expansion")
        return target

    def normalize(value: Any, expanding: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [normalize(item, expanding) for item in value]
        if not isinstance(value, dict):
            return value
        siblings = value.keys() - {"$ref", "default"}
        if "$ref" in value and siblings:
            # Validation siblings cannot safely be merged by dictionary overwrite.
            if siblings - annotations:
                raise ValueError("Unsupported schema reference validation siblings")
            ref = value["$ref"]
            target = resolve(ref)
            if ref in expanding or len(expanding) >= 32:
                raise ValueError("Cyclic or excessive schema reference annotation expansion")
            merged = {**target, **{key: item for key, item in value.items() if key in siblings}}
            return normalize(merged, (*expanding, ref))
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "default":
                continue
            if key in schema_maps and isinstance(item, dict):
                # Map keys are names (even "$ref" or "default"), not schema syntax.
                normalized[key] = {name: normalize(child, expanding)
                                   for name, child in item.items()}
            elif key in schema_values or key in schema_arrays:
                normalized[key] = normalize(item, expanding)
            else:
                # Annotations, literals and extensions can contain schema-shaped data.
                normalized[key] = deepcopy(item)
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["additionalProperties"] = False
            normalized["required"] = list(properties)
        return normalized

    return normalize(schema)
