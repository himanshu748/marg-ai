from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from eval.eval_agent import build_scenario
from marg.agent.llm import (
    BedrockLLM,
    BedrockUnavailable,
    LLMResponse,
    MockLLM,
    ToolCall,
    default_llm,
    provider_status,
)
from marg.agent.loop import run_agent
from marg.agent.tools import ToolContext, ToolSet
from marg.store.local import LocalStore


def context(tmp_path: Path, scenario: str = "S2") -> ToolContext:
    return ToolContext(
        result=build_scenario(scenario).result,
        store=LocalStore(tmp_path), detector=object(), keyframe_dir=tmp_path,
    )


def test_explicit_provider_never_silently_falls_back(monkeypatch):
    monkeypatch.setenv("MARG_LLM", "mock")
    assert isinstance(default_llm("bedrock"), BedrockLLM)
    assert isinstance(default_llm("mock"), MockLLM)
    with pytest.raises(ValueError, match="provider"):
        default_llm("beedrock")


def test_offline_provenance_and_policy_audit(tmp_path):
    tools = context(tmp_path)
    run = run_agent(tools.result, MockLLM(), tools)
    assert run.status == "completed"
    assert run.audit["passed"]
    assert run.provider == {"kind": "mock", "model_id": None, "region": None, "live_inference": False}
    assert run.to_dict()["run_id"]
    assert run.to_dict()["completed_at"]


def test_disabled_bedrock_never_creates_sdk_client(tmp_path, monkeypatch):
    monkeypatch.delenv("MARG_BEDROCK_ENABLED", raising=False)
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: pytest.fail("No AWS calls are permitted"))
    tools = context(tmp_path)
    run = run_agent(tools.result, BedrockLLM(), tools)
    assert run.status == "unavailable"
    assert run.error["code"] == "BedrockDisabled"
    assert run.provider["kind"] == "bedrock"
    assert run.provider["live_inference"] is False
    assert run.work_orders == []
    assert run.trace.entries[-1].tool == "provider_error"
    assert provider_status("bedrock")["readiness"] == "disabled"


def test_bedrock_reuses_bounded_client_and_records_actual_inference(monkeypatch):
    monkeypatch.setenv("MARG_BEDROCK_ENABLED", "1")
    created = []
    requests = []

    class Client:
        def converse(self, **kwargs):
            requests.append(kwargs)
            return {"output": {"message": {"content": [{"text": "Review ready."}]}}}

    def make_client(*args, **kwargs):
        created.append(kwargs)
        return Client()

    monkeypatch.setattr("boto3.client", make_client)
    llm = BedrockLLM(model_id="test-model", region="us-east-1")
    llm.converse([], [])
    llm.converse([], [])
    assert len(created) == 1
    config = created[0]["config"]
    assert config.connect_timeout == 5
    assert config.read_timeout == 60
    assert config.retries == {"mode": "adaptive", "total_max_attempts": 2}
    assert requests[0]["inferenceConfig"]["maxTokens"] == 1024
    assert llm.live_inference is True


def test_bedrock_access_error_is_classified_without_raw_cloud_detail(monkeypatch):
    monkeypatch.setenv("MARG_BEDROCK_ENABLED", "1")

    class Client:
        def converse(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "sensitive account detail"}}, "Converse")

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: Client())
    with pytest.raises(BedrockUnavailable) as failure:
        BedrockLLM().converse([], [])
    assert failure.value.code == "AccessDeniedException"
    assert failure.value.retryable is False
    assert "sensitive account detail" not in str(failure.value)


def test_partial_review_is_retained_and_blocked_when_provider_fails(tmp_path):
    class InterruptingProvider:
        calls = 0

        def converse(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(tool_calls=[ToolCall("draft_work_order", {"segment_id": 0, "priority": "high", "summary": "Repair evidence", "reason": "Severity 5"})])
            raise BedrockUnavailable("Access unavailable", "AccessDeniedException")

    tools = context(tmp_path)
    run = run_agent(tools.result, InterruptingProvider(), tools)
    assert run.status == "unavailable"
    assert len(run.work_orders) == 1
    assert run.work_orders[0]["status"] == "review_blocked"
    assert run.provider["kind"] == "custom"
    assert [entry.tool for entry in run.trace.entries] == ["draft_work_order", "provider_error"]


def test_missing_approval_fails_mandatory_audit(tmp_path):
    class UngatedProvider:
        def converse(self, messages, tools):
            return LLMResponse(tool_calls=[
                ToolCall("draft_work_order", {"segment_id": 0, "priority": "high", "summary": "Repair", "reason": "Critical damage"}),
                ToolCall("finalize", {"summary": "Finished"}),
            ])

    tools = context(tmp_path)
    run = run_agent(tools.result, UngatedProvider(), tools)
    assert run.status == "failed"
    assert run.error["code"] == "PolicyAuditFailed"
    assert run.work_orders[0]["status"] == "review_blocked"


def test_tool_schema_rejects_invalid_inputs_without_side_effects(tmp_path):
    tools = ToolSet(context(tmp_path))
    assert "error" in tools.call("dismiss_instance", {"instance_id": True, "reason": "bad"})
    assert "error" in tools.call("draft_work_order", {"segment_id": 0})
    assert "error" in tools.call("finalize", {"summary": "", "extra": "bad"})
    assert not tools.context.work_orders
    assert not tools.context.dismissals
    assert not tools.finalized


def test_nonexistent_and_dismissed_evidence_cannot_create_orders(tmp_path):
    tools = ToolSet(context(tmp_path))
    assert "error" in tools.draft_work_order(9, "high", "Invented segment")
    assert "error" in tools.request_resurvey(9, "Invented segment")
    assert "error" in tools.dismiss_instance(9, "Invented instance")
    tools.dismiss_instance(1, "unconfirmed")
    assert "error" in tools.draft_work_order(0, "high", "Dismissed evidence")
    assert not tools.context.work_orders


def test_low_confidence_requires_confirmed_inspection_and_priority_cannot_bypass_approval(tmp_path):
    tools = ToolSet(context(tmp_path, "S1"))
    assert "error" in tools.draft_work_order(0, "medium", "Unverified evidence")
    tools.inspections[1] = {"confirmed": False}
    assert "error" in tools.draft_work_order(0, "medium", "Failed inspection")
    tools.inspections[1] = {"confirmed": True}
    assert "error" in tools.draft_work_order(0, "low", "Downplay severity")
    order = tools.draft_work_order(0, "medium", "Verified evidence")
    assert order["instance_ids"] == [1]
    assert tools.draft_work_order(0, "medium", "Duplicate")["work_order_id"] == order["work_order_id"]
    assert len(tools.context.work_orders) == 1
    assert "error" in tools.dismiss_instance(1, "Now dismissed")


@pytest.mark.parametrize("scenario", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_all_policy_scenarios_with_fixture_inspection(tmp_path, scenario):
    from eval.eval_agent import evaluate_run

    tools = ToolSet(context(tmp_path, scenario))
    # A deterministic tool fixture verifies orchestration, not detector accuracy.
    tools.inspect_roi = lambda instance_id: {"confirmed": True, "detections": [{"class": "D40", "conf": 0.9}]}
    run = run_agent(tools.context.result, MockLLM(), tools)
    assert evaluate_run(build_scenario(scenario), run)
    if scenario == "S6":
        assert run.status == "failed"  # Budget-limited evidence is not a complete policy review.
        assert not run.audit["passed"]
    else:
        assert run.audit["passed"]


def test_auditor_rejects_invalid_or_dismissed_work_order_evidence(tmp_path):
    from marg.agent.auditor import audit

    tools = context(tmp_path)
    run = run_agent(tools.result, MockLLM(), tools)
    run.work_orders[0]["segment_id"] = 999
    assert not audit(tools.result, run).passed
    run.work_orders[0]["segment_id"] = 0
    run.dismissals.append({"instance_id": 1})
    assert not audit(tools.result, run).passed


def test_provider_that_never_uses_tools_stops_after_three_responses(tmp_path):
    class TextOnlyProvider:
        calls = 0

        def converse(self, messages, tools):
            self.calls += 1
            assert messages[-1]["role"] == "user"
            return LLMResponse(text="I will review this evidence.")

    provider = TextOnlyProvider()
    tools = context(tmp_path)
    run = run_agent(tools.result, provider, tools)
    assert provider.calls == 3
    assert run.status == "failed"
    assert run.error["code"] == "NoToolAction"


def test_malformed_tool_arguments_cannot_crash_final_audit(tmp_path):
    class MalformedProvider:
        def converse(self, messages, tools):
            return LLMResponse(tool_calls=[
                ToolCall("compare_frames", {"instance_id": {"unexpected": "object"}}),
                ToolCall("finalize", {"summary": "Review incomplete"}),
            ])

    tools = context(tmp_path)
    run = run_agent(tools.result, MalformedProvider(), tools)
    assert run.status == "failed"
    assert run.trace.entries[0].output.get("error")
    assert not run.work_orders
