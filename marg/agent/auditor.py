from dataclasses import dataclass, field

from .loop import AgentRun


@dataclass(slots=True)
class AuditRule:
    name: str
    passed: bool
    offending_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AuditReport:
    passed: bool
    rules: list[AuditRule]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "rules": [
                {
                    "name": rule.name,
                    "passed": rule.passed,
                    "offending_ids": rule.offending_ids,
                }
                for rule in self.rules
            ],
        }


def audit(result: object, run: AgentRun) -> AuditReport:
    instances = getattr(result, "instances", [])
    segments = {segment.id: segment for segment in getattr(result, "segments", [])}
    entries = run.trace.entries
    rule_one_offenders: list[str] = []
    for work_order in run.work_orders:
        segment = segments.get(int(work_order.get("segment_id", -1)))
        if segment is None:
            continue
        draft_step = next(
            (
                entry.step
                for entry in entries
                if entry.tool == "draft_work_order"
                and int(entry.input.get("segment_id", -1)) == segment.id
            ),
            len(entries),
        )
        for instance in instances:
            if instance.id not in segment.instance_ids or instance.fused_conf >= 0.55:
                continue
            inspected = any(
                entry.tool == "inspect_roi"
                and int(entry.input.get("instance_id", -1)) == instance.id
                and entry.step < draft_step
                for entry in entries
            )
            if not inspected:
                rule_one_offenders.append(str(instance.id))
    rule_three_offenders: list[str] = []
    for work_order in run.work_orders:
        priority = str(work_order.get("priority", ""))
        if priority not in {"high", "medium"}:
            continue
        order_id = str(work_order.get("work_order_id", ""))
        approved = bool(work_order.get("approval_requested")) or any(
            entry.tool == "request_human_approval"
            and str(entry.input.get("work_order_id")) == order_id
            for entry in entries
        )
        if not approved:
            rule_three_offenders.append(order_id)
    finalize_count = sum(1 for entry in entries if entry.tool == "finalize")
    rules = [
        AuditRule("rule_1_low_confidence_inspection", not rule_one_offenders, rule_one_offenders),
        AuditRule("rule_3_approval", not rule_three_offenders, rule_three_offenders),
        AuditRule(
            "rule_4_5_finalize_budget",
            finalize_count == 1 and run.tool_call_count <= 25,
            [f"finalize_count={finalize_count}", f"tool_calls={run.tool_call_count}"]
            if finalize_count != 1 or run.tool_call_count > 25
            else [],
        ),
    ]
    return AuditReport(passed=all(rule.passed for rule in rules), rules=rules)
