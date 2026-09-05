import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from marg.store.local import LocalStore
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.models import SurveyResult

from .llm import BedrockLLM, LLM, LLMResponse, MockLLM, ToolCall, default_llm
from .prompts import FINALIZE_INSTRUCTION, SYSTEM_PROMPT
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

    def to_dict(self) -> dict[str, object]:
        return {
            "trace": [asdict(entry) for entry in self.trace.entries],
            "summary": self.summary,
            "work_orders": self.work_orders,
            "resurveys": self.resurveys,
            "dismissals": self.dismissals,
            "forced_finalize": self.forced_finalize,
            "tool_call_count": self.tool_call_count,
        }


def run_agent(
    result: SurveyResult,
    llm: LLM,
    tools: ToolContext | ToolSet,
    max_tool_calls: int = 25,
) -> AgentRun:
    registry = tools if isinstance(tools, ToolSet) else ToolSet(tools)
    if isinstance(llm, MockLLM):
        llm.bind(result)
    trace = Trace()
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"{SYSTEM_PROMPT}\n\nSURVEY RESULT:\n{result.model_dump_json()}"
                }
            ],
        }
    ]
    tool_call_count = 0
    forced_finalize = False
    response = LLMResponse()
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
            messages.append({"role": "assistant", "content": [{"text": response.text}]})
            continue
        call = response.tool_calls[0]
        output = _execute_call(registry, call, trace, llm)
        tool_call_count += 1
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": call.call_id or f"call-{tool_call_count}",
                            "name": call.name,
                            "input": call.arguments,
                        }
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": call.call_id or f"call-{tool_call_count}",
                            "content": [{"json": output}],
                        }
                    }
                ],
            }
        )
        if registry.finalized:
            break
    if not registry.finalized:
        registry.finalize("budget exhausted")
        forced_finalize = True
    return AgentRun(
        trace=trace,
        summary=registry.summary,
        work_orders=registry.context.work_orders,
        resurveys=registry.context.resurveys,
        dismissals=registry.context.dismissals,
        forced_finalize=forced_finalize,
        tool_call_count=tool_call_count,
    )


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
    run.trace.write_jsonl(output_dir / "trace.jsonl")
    (output_dir / "agent_result.json").write_text(
        json.dumps(run.to_dict(), indent=2), encoding="utf-8"
    )
    print(json.dumps(run.to_dict(), indent=2))


if __name__ == "__main__":
    main()
