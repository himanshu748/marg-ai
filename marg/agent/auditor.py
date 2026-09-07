from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def audit(result: object, run: "AgentRun") -> AuditReport:
    instances = getattr(result, "instances", [])
    segments = {segment.id: segment for segment in getattr(result, "segments", [])}
    entries = run.trace.entries
    rule_one_offenders: list[str] = []
    evidence_offenders: list[str] = []
    resurvey_offenders: list[str] = []
    coverage_offenders: list[str] = []
    dismissed = {item.get("instance_id") for item in run.dismissals}
    by_id = {item.id: item for item in instances}
    for work_order in run.work_orders:
        segment_id = work_order.get("segment_id")
        segment = segments.get(segment_id) if isinstance(segment_id, int) else None
        order_id = str(work_order.get("work_order_id", ""))
        if segment is None:
            evidence_offenders.append(order_id)
            continue
        instance_ids = work_order.get("instance_ids", segment.instance_ids)
        if not isinstance(instance_ids, list) or not instance_ids or any(
            not isinstance(item, int) or item not in by_id or item not in segment.instance_ids or item in dismissed
            for item in instance_ids
        ):
            evidence_offenders.append(order_id)
            continue
        draft_step = next(
            (
                entry.step
                for entry in entries
                if entry.tool == "draft_work_order"
                and entry.input.get("segment_id") == segment.id
                and not entry.output.get("error")
            ),
            len(entries),
        )
        for instance in instances:
            if instance.id not in instance_ids or instance.fused_conf >= 0.55:
                continue
            inspected = any(
                entry.tool == "inspect_roi"
                and entry.input.get("instance_id") == instance.id
                and entry.output.get("confirmed") is True
                and entry.step < draft_step
                for entry in entries
            )
            if not inspected:
                rule_one_offenders.append(str(instance.id))
    rule_three_offenders: list[str] = []
    for work_order in run.work_orders:
        priority = str(work_order.get("priority", ""))
        ids = work_order.get("instance_ids", [])
        evidence = [by_id[item] for item in ids if isinstance(item, int) and item in by_id] if isinstance(ids, list) else []
        needs_approval = priority in {"high", "medium"} or any(item.severity >= 4 for item in evidence) or sum(item.class_name == "D40" for item in evidence) >= 3
        if not needs_approval:
            continue
        order_id = str(work_order.get("work_order_id", ""))
        approved = bool(work_order.get("approval_requested")) and any(
            entry.tool == "request_human_approval"
            and str(entry.input.get("work_order_id")) == order_id
            and entry.output.get("status") == "pending_approval"
            for entry in entries
        )
        if not approved:
            rule_three_offenders.append(order_id)
    finalize_count = sum(1 for entry in entries if entry.tool == "finalize" and entry.output.get("status") == "finalized")
    for entry in entries:
        if entry.tool != "compare_frames":
            continue
        instance_id = entry.input.get("instance_id")
        instance = by_id.get(instance_id) if isinstance(instance_id, int) else None
        if instance is None or instance.severity < 3:
            continue
        classes = entry.output.get("classes_seen", [])
        if not isinstance(classes, list) or len({str(item) for item in classes}) <= 1:
            continue
        for segment in segments.values():
            if instance.id in segment.instance_ids and not any(item.get("segment_id") == segment.id for item in run.resurveys):
                resurvey_offenders.append(str(segment.id))
    confirmed = {
        entry.input.get("instance_id") for entry in entries
        if entry.tool == "inspect_roi" and entry.output.get("confirmed") is True
    }
    for segment in segments.values():
        actionable = [item for item in instances if item.id in segment.instance_ids and item.id not in dismissed and (item.fused_conf >= 0.55 or item.id in confirmed)]
        orders = [item for item in run.work_orders if item.get("segment_id") == segment.id]
        required = any(item.severity >= 4 for item in actionable) or sum(item.class_name == "D40" for item in actionable) >= 3 or (len(actionable) >= 2 and all(item.severity <= 2 for item in actionable))
        if len(orders) > 1 or (required and not orders):
            coverage_offenders.append(str(segment.id))
        for order in orders:
            if order.get("priority") == "low" and (len(actionable) < 2 or any(item.severity > 2 for item in actionable)):
                coverage_offenders.append(str(segment.id))
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
        AuditRule("valid_actionable_evidence", not evidence_offenders, evidence_offenders),
        AuditRule("rule_2_disagreement_resurvey", not resurvey_offenders, resurvey_offenders),
        AuditRule("required_segment_actions", not coverage_offenders, coverage_offenders),
    ]
    return AuditReport(passed=all(rule.passed for rule in rules), rules=rules)
