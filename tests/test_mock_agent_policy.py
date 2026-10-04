from pathlib import Path

import pytest

from eval.eval_agent import build_scenario, evaluate_run, run_scenario
from marg.agent.llm import MockLLM
from marg.agent.loop import run_agent
from marg.agent.tools import ToolContext, ToolSet
from marg.store.local import LocalStore


def test_mock_agent_policy_s1_to_s5(tmp_path: Path) -> None:
    for name in ["S1", "S2", "S3", "S4", "S5"]:
        scenario = build_scenario(name)
        run = run_scenario(scenario, tmp_path)
        assert evaluate_run(scenario, run), (name, [entry.tool for entry in run.trace.entries])


@pytest.mark.parametrize(
    ("severities", "priority"),
    [([3], None), ([3, 3], None), ([3, 3, 3], "medium"),
     ([4], "medium"), ([5], "high"), ([4, 5], "high"),
     ([2, 2, 2], "medium")],
)
def test_segment_escalation_matches_tool_policy(tmp_path, severities, priority):
    result = build_scenario("S2").result
    template = result.instances[0]
    result.instances = [
        template.model_copy(update={"id": index, "severity": severity})
        for index, severity in enumerate(severities, start=1)
    ]
    result.segments[0].instance_ids = [item.id for item in result.instances]

    def registry(root):
        return ToolSet(ToolContext(result, LocalStore(root), object(), tmp_path))

    direct = registry(tmp_path / "direct").draft_work_order(
        0, priority or "medium", "Repair road", "Confirmed observations"
    )
    assert bool(direct.get("error")) == (priority is None)

    run = run_agent(result, MockLLM(), registry(tmp_path / "agent"))
    tools = [entry.tool for entry in run.trace.entries if entry.tool != "model"]
    assert run.status == "completed"
    assert run.audit["passed"]
    assert not run.forced_finalize
    assert not any(entry.output.get("error") for entry in run.trace.entries)
    assert tools == (
        ["draft_work_order", "request_human_approval", "finalize"]
        if priority else ["finalize"]
    )
    assert [order["priority"] for order in run.work_orders] == (
        [priority] if priority else []
    )


def test_unconfirmed_pothole_does_not_count_toward_escalation(tmp_path):
    result = build_scenario("S2").result
    template = result.instances[0]
    result.instances = [
        template.model_copy(update={"id": index, "severity": 3})
        for index in range(1, 4)
    ]
    result.instances[-1].fused_conf = 0.4
    result.segments[0].instance_ids = [1, 2, 3]
    registry = ToolSet(ToolContext(result, LocalStore(tmp_path), object(), tmp_path))
    registry.inspect_roi = lambda instance_id: {"instance_id": instance_id, "confirmed": False}
    run = run_agent(result, MockLLM(), registry)
    assert run.status == "completed"
    assert not run.work_orders
    assert [entry.tool for entry in run.trace.entries if entry.tool != "model"] == [
        "inspect_roi", "compare_frames", "dismiss_instance", "finalize"
    ]
