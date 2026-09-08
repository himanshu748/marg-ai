import os
from dataclasses import dataclass, field
from typing import Protocol

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from marg.vision.models import SurveyResult

from .prompts import FINALIZE_INSTRUCTION, SYSTEM_PROMPT


class BedrockUnavailable(RuntimeError):
    """Raised when Bedrock cannot service a Converse request."""

    def __init__(self, message: str, code: str = "BedrockUnavailable", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "retryable": self.retryable}


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, object]
    call_id: str = ""


@dataclass(slots=True)
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def converse(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> LLMResponse:
        ...


class BedrockLLM:
    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get(
            "MARG_BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0"
        )
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.live_inference = False
        self._client = None
        self.config = Config(
            connect_timeout=_bounded_integer("MARG_BEDROCK_CONNECT_TIMEOUT", 5, 1, 30),
            read_timeout=_bounded_integer("MARG_BEDROCK_READ_TIMEOUT", 60, 1, 300),
            retries={
                "mode": "adaptive",
                "total_max_attempts": _bounded_integer("MARG_BEDROCK_MAX_ATTEMPTS", 2, 1, 5),
            },
        )
        self.max_tokens = _bounded_integer("MARG_BEDROCK_MAX_TOKENS", 1024, 128, 4096)

    def converse(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> LLMResponse:
        if os.environ.get("MARG_BEDROCK_ENABLED") != "1":
            raise BedrockUnavailable(
                "Bedrock review is disabled until model access is ready. Use an explicit offline review or enable Bedrock in deployment configuration.",
                "BedrockDisabled",
            )
        import boto3

        try:
            if self._client is None:
                self._client = boto3.client(
                    "bedrock-runtime", region_name=self.region, config=self.config
                )
            response = self._client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=messages,
                toolConfig={"tools": tools},
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0},
            )
        except (ClientError, BotoCoreError) as exc:
            code = type(exc).__name__
            if isinstance(exc, ClientError):
                code = str(exc.response.get("Error", {}).get("Code", code))
            retryable = code in {
                "ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException",
                "InternalServerException", "ReadTimeoutError", "ConnectTimeoutError",
                "EndpointConnectionError", "ConnectionClosedError",
            }
            raise BedrockUnavailable(
                f"Bedrock review is unavailable ({code}). Check model access, billing and quotas before retrying.",
                code=code,
                retryable=retryable,
            ) from exc
        self.live_inference = True
        output = response.get("output", {})
        if not isinstance(output, dict):
            raise BedrockUnavailable("Bedrock returned an invalid response.", "InvalidResponse")
        message = output.get("message", {})
        if not isinstance(message, dict):
            raise BedrockUnavailable("Bedrock returned an invalid response.", "InvalidResponse")
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        content = message.get("content", [])
        if not isinstance(content, list):
            content = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            tool_use = block.get("toolUse")
            if isinstance(tool_use, dict):
                name = tool_use.get("name")
                arguments = tool_use.get("input", {})
                call_id = tool_use.get("toolUseId", "")
                if isinstance(name, str) and isinstance(arguments, dict):
                    calls.append(
                        ToolCall(
                            name=name,
                            arguments={str(key): value for key, value in arguments.items()},
                            call_id=str(call_id),
                        )
                    )
        if not text_parts and not calls:
            raise BedrockUnavailable("Bedrock returned no usable review content.", "EmptyResponse")
        return LLMResponse(text="\n".join(text_parts), tool_calls=calls)


class MockLLM:
    def __init__(self, result: SurveyResult | None = None) -> None:
        self.result = result
        self.inspected: set[int] = set()
        self.confirmed: set[int] = set()
        self.compared: set[int] = set()
        self.disagreements: set[int] = set()
        self.resurveyed: set[int] = set()
        self.work_ordered: set[int] = set()
        self.approved: set[str] = set()
        self.dismissed: set[int] = set()
        self.finalized = False

    def bind(self, result: SurveyResult) -> None:
        self.result = result

    def observe(self, call: ToolCall, output: dict[str, object]) -> None:
        if output.get("error"):
            return
        if call.name == "inspect_roi":
            instance_id = int(call.arguments["instance_id"])
            self.inspected.add(instance_id)
            if bool(output.get("confirmed")):
                self.confirmed.add(instance_id)
        elif call.name == "compare_frames":
            instance_id = int(call.arguments["instance_id"])
            self.compared.add(instance_id)
            classes = output.get("classes_seen", [])
            if isinstance(classes, list) and len({str(item) for item in classes}) > 1:
                self.disagreements.add(instance_id)
        elif call.name == "request_resurvey":
            self.resurveyed.add(int(call.arguments["segment_id"]))
        elif call.name == "draft_work_order":
            self.work_ordered.add(int(call.arguments["segment_id"]))
        elif call.name == "request_human_approval":
            self.approved.add(str(call.arguments["work_order_id"]))
        elif call.name == "dismiss_instance":
            self.dismissed.add(int(call.arguments["instance_id"]))
        elif call.name == "finalize":
            self.finalized = True

    def converse(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> LLMResponse:
        if self.result is None:
            return LLMResponse(tool_calls=[ToolCall("finalize", {"summary": "No survey result."})])
        if messages and any(
            isinstance(content, dict)
            and content.get("text") == FINALIZE_INSTRUCTION
            for message in messages
            for content in (message.get("content", []) if isinstance(message.get("content", []), list) else [])
        ):
            return LLMResponse(
                text="Budget reached; finalizing the evidence collected.",
                tool_calls=[ToolCall("finalize", {"summary": "Budget reached; review stopped at the tool limit."})],
            )
        for instance in self.result.instances:
            if instance.fused_conf < 0.55 and instance.id not in self.inspected:
                return self._call(
                    f"Inspect low-confidence instance {instance.id} under policy rule 1.",
                    "inspect_roi",
                    {"instance_id": instance.id},
                )
        for instance in self.result.instances:
            if instance.fused_conf < 0.55 and instance.id in self.inspected and instance.id not in self.confirmed and instance.id not in self.compared:
                return self._call(
                    f"Compare frames for uncertain instance {instance.id} under policy rule 1.",
                    "compare_frames",
                    {"instance_id": instance.id},
                )
            if instance.id in self.compared and instance.id not in self.confirmed and instance.id not in self.dismissed:
                return self._call(
                    f"Dismiss unconfirmed instance {instance.id} under policy rule 1.",
                    "dismiss_instance",
                    {"instance_id": instance.id, "reason": "unconfirmed"},
                )
            classes = instance.frame_classes
            if (
                instance.severity >= 3
                and len(set(classes)) > 1
                and instance.id not in self.compared
            ):
                return self._call(
                    f"Compare class disagreement for instance {instance.id} under policy rule 2.",
                    "compare_frames",
                    {"instance_id": instance.id},
                )
            if instance.id in self.disagreements and instance.severity >= 3:
                segment_id = self._segment_for(instance.id)
                if segment_id not in self.resurveyed:
                    return self._call(
                        f"Request a re-survey for segment {segment_id} under policy rule 2.",
                        "request_resurvey",
                        {"segment_id": segment_id, "reason": f"Class disagreement for instance {instance.id}."},
                    )
            confirmed = instance.fused_conf >= 0.55 or instance.id in self.confirmed
            if confirmed and (instance.severity >= 4 or (instance.class_name == "D40" and instance.severity >= 3)):
                segment_id = self._segment_for(instance.id)
                if segment_id not in self.work_ordered:
                    priority = "high" if instance.severity >= 5 else "medium"
                    return self._call(
                        f"Escalate segment {segment_id} under policy rule 3.",
                        "draft_work_order",
                        {
                            "segment_id": segment_id,
                            "priority": priority,
                            "summary": f"Road damage severity {instance.severity}.",
                            "reason": (
                                f"{instance.class_name} instance {instance.id} has "
                                f"severity {instance.severity} and fused confidence "
                                f"{instance.fused_conf:.2f}."
                            ),
                        },
                    )
        for segment in self.result.segments:
            if segment.id in self.work_ordered:
                continue
            segment_instances = [item for item in self.result.instances if item.id in segment.instance_ids]
            if len(segment_instances) >= 2 and all(item.severity <= 2 for item in segment_instances):
                return self._call(
                    f"Draft one low-priority order for segment {segment.id} under policy rule 4.",
                    "draft_work_order",
                    {
                        "segment_id": segment.id,
                        "priority": "low",
                        "summary": "Multiple low-severity road defects.",
                        "reason": (
                            f"Segment {segment.id} contains {len(segment_instances)} "
                            "low-severity observations."
                        ),
                    },
                )
        for work_order in self._drafts(messages):
            work_order_id = str(work_order.get("work_order_id", ""))
            priority = str(work_order.get("priority", ""))
            if priority in {"high", "medium"} and work_order_id not in self.approved:
                return self._call(
                    f"Request approval for work order {work_order_id} under policy rule 3.",
                    "request_human_approval",
                    {"work_order_id": work_order_id},
                )
        return self._call(
            "Finalize the survey review under policy rule 5.",
            "finalize",
            {"summary": "Review complete; uncertainty and escalations were recorded."},
        )

    def _call(self, text: str, name: str, arguments: dict[str, object]) -> LLMResponse:
        return LLMResponse(text=text, tool_calls=[ToolCall(name=name, arguments=arguments)])

    def _segment_for(self, instance_id: int) -> int:
        if self.result is None:
            return -1
        for segment in self.result.segments:
            if instance_id in segment.instance_ids:
                return segment.id
        return -1

    def _drafts(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        drafts: list[dict[str, object]] = []
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                result = block.get("toolResult")
                if isinstance(result, dict):
                    value = result.get("content", [])
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and isinstance(item.get("json"), dict):
                                data = item["json"]
                                if data.get("work_order_id"):
                                    drafts.append(data)
        return drafts


def _bounded_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def provider_kind(kind: str | None = None) -> str:
    selected = kind or os.environ.get("MARG_AGENT_LLM") or os.environ.get("MARG_LLM")
    selected = selected or ("bedrock" if os.environ.get("MARG_BEDROCK_MODEL_ID") else "mock")
    if selected not in {"mock", "bedrock"}:
        raise ValueError("Agent provider must be mock or bedrock")
    return selected


def provider_status(kind: str | None = None) -> dict[str, object]:
    """Describe configuration without invoking a model or probing credentials."""
    selected = provider_kind(kind)
    return {
        "kind": selected,
        "model_id": os.environ.get("MARG_BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0") if selected == "bedrock" else None,
        "region": os.environ.get("AWS_REGION", "us-east-1") if selected == "bedrock" else None,
        "live_inference": False,
        "readiness": ("unverified" if os.environ.get("MARG_BEDROCK_ENABLED") == "1" else "disabled") if selected == "bedrock" else "offline",
    }


def provider_details(llm: LLM) -> dict[str, object]:
    if isinstance(llm, MockLLM):
        return {"kind": "mock", "model_id": None, "region": None, "live_inference": False}
    if isinstance(llm, BedrockLLM):
        return {"kind": "bedrock", "model_id": llm.model_id, "region": llm.region, "live_inference": llm.live_inference}
    return {"kind": "custom", "model_id": None, "region": None, "live_inference": False}


def default_llm(kind: str | None = None) -> LLM:
    if provider_kind(kind) == "mock":
        return MockLLM()
    return BedrockLLM()
