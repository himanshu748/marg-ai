from eval.eval_agent import build_scenario
from marg.agent.auditor import audit
from marg.agent.loop import AgentRun
from marg.agent.trace import Trace


def test_auditor_catches_rule_one_and_three() -> None:
    scenario = build_scenario("S1")
    trace = Trace()
    trace.append("draft_work_order", {"segment_id": 0, "priority": "high", "summary": "bad"}, {"work_order_id": "wo-1"}, 0.1)
    run = AgentRun(
        trace=trace,
        summary="bad",
        work_orders=[{"work_order_id": "wo-1", "segment_id": 0, "priority": "high", "approval_requested": False}],
        tool_call_count=1,
    )
    report = audit(scenario.result, run)
    assert report.passed is False
    assert report.rules[0].passed is False
    assert report.rules[1].passed is False
