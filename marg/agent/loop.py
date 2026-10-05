import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from marg.store.local import LocalStore, atomic_write_json
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.models import SurveyResult

from .llm import (
    LLM,
    BedrockUnavailable,
    LLMResponse,
    MockLLM,
    ToolCall,
    default_llm,
    provider_details,
)
from .prompts import FINALIZE_INSTRUCTION
from .tools import TOOL_SPECS, ToolContext, ToolSet
from .trace import Trace


@dataclass(slots=True)
class AgentRun:
    trace: Trace
    summary: str
    work_orders: list[dict[str, object]] = field(default_factory=list)
    resurveys: list[dict[str, object]] = field(default_factory=list)
    dismissals: list[dict[str, object]] = field(default_factory=list)
    forced_finalize: bool = False
    tool_call_count: int = 0
    provider: dict[str, object] = field(default_factory=dict)
    status: str = "completed"
    error: dict[str, object] | None = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    audit: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "trace": [asdict(entry) for entry in self.trace.entries],
            "summary": self.summary,
            "work_orders": self.work_orders,
            "resurveys": self.resurveys,
            "dismissals": self.dismissals,
            "forced_finalize": self.forced_finalize,
            "tool_call_count": self.tool_call_count,
            "provider": self.provider,
            "status": self.status,
            "error": self.error,
            "run_id": self.run_id,
            "completed_at": self.completed_at,
            "audit": self.audit,
        }


def run_agent(
    result: SurveyResult,
    llm: LLM,
    tools: ToolContext | ToolSet,
    max_tool_calls: int = 25,
) -> AgentRun:
    if max_tool_calls < 1 or max_tool_calls > 25:
        raise ValueError("max_tool_calls must be between 1 and 25")
    registry = tools if isinstance(tools, ToolSet) else ToolSet(tools)
    if isinstance(llm, MockLLM):
        llm.bind(result)
    trace = Trace()
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"SURVEY RESULT:\n{result.model_dump_json()}"
                }
            ],
        }
    ]
    tool_call_count = 0
    forced_finalize = False
    no_action_count = 0
    status = "completed"
    error = None
    try:
        for _ in range(max_tool_calls + 10):
            if tool_call_count >= max_tool_calls - 1 and not registry.finalized:
                response = _finalize_response(llm, messages, trace)
                if response.tool_calls:
                    call = response.tool_calls[0]
                    if call.name == "finalize":
                        _execute_call(registry, call, trace, llm)
                        tool_call_count += 1
                if not registry.finalized:
                    registry.finalize("budget exhausted")
                    forced_finalize = True
                break
            started = time.perf_counter()
            response = llm.converse(messages, TOOL_SPECS)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if response.text:
                trace.append("model", {}, {"text": response.text}, latency_ms, response.text)
            if not response.tool_calls:
                no_action_count += 1
                if no_action_count >= 3:
                    status = "failed"
                    error = {"code": "NoToolAction", "message": "The provider did not complete an actionable review.", "retryable": True}
                    break
                if response.text:
                    messages.append({"role": "assistant", "content": [{"text": response.text}]})
                messages.append({"role": "user", "content": [{"text": "Continue using the review tools. Call finalize when the evidence review is complete."}]})
                continue
            no_action_count = 0
            remaining = max_tool_calls - 1 - tool_call_count
            calls = response.tool_calls[:remaining]
            outputs: list[tuple[ToolCall, dict[str, object]]] = []
            for call in calls:
                output = _execute_call(registry, call, trace, llm)
                tool_call_count += 1
                outputs.append((call, output))
                if registry.finalized:
                    break
            _append_tool_exchange(messages, outputs, tool_call_count)
            if registry.finalized:
                break
        if not registry.finalized:
            registry.finalize("budget exhausted")
            forced_finalize = True
    except BedrockUnavailable as exc:
        status = "unavailable"
        error = exc.to_dict()
        trace.append("provider_error", {}, error, 0.0)
        registry.summary = "Agent review unavailable. Vision evidence is preserved for review and retry."
    run = AgentRun(
        trace=trace,
        summary=registry.summary,
        work_orders=registry.context.work_orders,
        resurveys=registry.context.resurveys,
        dismissals=registry.context.dismissals,
        forced_finalize=forced_finalize,
        tool_call_count=tool_call_count,
        provider=provider_details(llm),
        status=status,
        error=error,
    )
    from .auditor import audit

    run.audit = audit(result, run).to_dict()
    if run.status == "completed" and not run.audit["passed"]:
        run.status = "failed"
        run.error = {"code": "PolicyAuditFailed", "message": "Review did not satisfy the evidence and approval policy.", "retryable": True}
    if run.status != "completed":
        for work_order in run.work_orders:
            work_order["status"] = "review_blocked"
    return run


def _execute_call(
    registry: ToolSet,
    call: ToolCall,
    trace: Trace,
    llm: LLM,
) -> dict[str, object]:
    started = time.perf_counter()
    output = registry.call(call.name, call.arguments)
    latency_ms = (time.perf_counter() - started) * 1000.0
    trace.append(call.name, call.arguments, output, latency_ms)
    if isinstance(llm, MockLLM):
        llm.observe(call, output)
    return output


def _append_tool_exchange(
    messages: list[dict[str, object]],
    outputs: list[tuple[ToolCall, dict[str, object]]],
    tool_call_count: int,
) -> None:
    tool_uses: list[dict[str, object]] = []
    tool_results: list[dict[str, object]] = []
    for offset, (call, output) in enumerate(outputs):
        call_id = call.call_id or f"call-{tool_call_count - len(outputs) + offset + 1}"
        tool_uses.append(
            {
                "toolUse": {
                    "toolUseId": call_id,
                    "name": call.name,
                    "input": call.arguments,
                }
            }
        )
        tool_results.append(
            {
                "toolResult": {
                    "toolUseId": call_id,
                    "content": [{"json": output}],
                    "status": "error" if output.get("error") else "success",
                }
            }
        )
    messages.append({"role": "assistant", "content": tool_uses})
    messages.append({"role": "user", "content": tool_results})


def _finalize_response(llm: LLM, messages: list[dict[str, object]], trace: Trace) -> LLMResponse:
    started = time.perf_counter()
    final_messages = messages + [{"role": "user", "content": [{"text": FINALIZE_INSTRUCTION}]}]
    response = llm.converse(final_messages, TOOL_SPECS)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response.text:
        trace.append("model", {}, {"text": response.text}, latency_ms, response.text)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MargAI survey agent.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--llm", choices=["mock", "bedrock"], default="mock")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = SurveyResult.model_validate_json(Path(args.result).read_text(encoding="utf-8"))
        config = VisionConfig()
        model_path = config.model_file(Path(__file__).resolve().parents[2])
        detector = DNNDetector(model_path, config)
        store = LocalStore(output_dir / "store")
        context = ToolContext(
            result=result,
            store=store,
            detector=detector,
            keyframe_dir=Path(args.result).parent / "keyframes",
        )
        llm = default_llm(args.llm)
        run = run_agent(result, llm, context)
    except BedrockUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    run.trace.write_jsonl(output_dir / "trace.jsonl")
    atomic_write_json(output_dir / "agent_result.json", run.to_dict())
    print(json.dumps(run.to_dict(), indent=2))
    if run.status != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
