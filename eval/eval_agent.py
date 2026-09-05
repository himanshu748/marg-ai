import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from marg.agent.llm import MockLLM
from marg.agent.loop import AgentRun, run_agent
from marg.agent.tools import ToolContext
from marg.store.local import LocalStore
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.models import Segment, SurveyInstance, SurveyResult

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "pothole_bengaluru.jpg"
MODEL = ROOT / "models" / "rdd_yolov8s.onnx"


@dataclass(slots=True)
class Scenario:
    name: str
    result: SurveyResult
    expected: str


def build_scenario(name: str) -> Scenario:
    base = {
        "video": "synthetic",
        "geo_source": "synthetic",
        "frames": 1,
        "processed_frames": 1,
        "keyframes": 1,
        "detections": {"D00": 0, "D10": 0, "D20": 0, "D40": 0},
        "keyframe_paths": [str(FIXTURE)],
    }
    if name == "S1":
        instances = [_instance(1, "D40", 0.40, 4, [0.40], ["D40"])]
        expected = "inspect_before_work_order"
    elif name == "S2":
        instances = [_instance(1, "D40", 0.90, 5, [0.90], ["D40"])]
        expected = "high_order_then_approval"
    elif name == "S3":
        instances = [
            _instance(1, "D00", 0.80, 2, [0.80], ["D00"]),
            _instance(2, "D00", 0.80, 2, [0.80], ["D00"]),
            _instance(3, "D00", 0.80, 2, [0.80], ["D00"]),
        ]
        expected = "one_low_order"
    elif name == "S4":
        instances = [_instance(1, "D20", 0.80, 3, [0.8, 0.8, 0.8, 0.8], ["D20", "D40", "D20", "D00"])]
        expected = "resurvey"
    elif name == "S5":
        instances = [_instance(1, "D10", 0.80, 1, [0.8], ["D10"])]
        expected = "no_work_order"
    elif name == "S6":
        instances = [_instance(index, "D40", 0.30, 4, [0.3], ["D40"]) for index in range(1, 31)]
        expected = "budgeted_finalize"
    else:
        raise ValueError(f"Unknown scenario: {name}")
    segment = Segment(
        id=0,
        instance_ids=[instance.id for instance in instances],
        lat=28.6139,
        lon=77.2090,
        radius_m=25.0,
    )
    result = SurveyResult(
        survey_id=name.lower(),
        instances=instances,
        segments=[segment],
        **base,
    )
    return Scenario(name=name, result=result, expected=expected)


def run_scenario(scenario: Scenario, root: Path) -> AgentRun:
    output = root / scenario.name
    output.mkdir(parents=True, exist_ok=True)
    context = ToolContext(
        result=scenario.result,
        store=LocalStore(output / "store"),
        detector=DNNDetector(MODEL, VisionConfig()),
        keyframe_dir=FIXTURE.parent,
    )
    return run_agent(scenario.result, MockLLM(), context)


def evaluate_run(scenario: Scenario, run: AgentRun) -> bool:
    names = [entry.tool for entry in run.trace.entries]
    if scenario.name == "S1":
        inspect = names.index("inspect_roi") if "inspect_roi" in names else 999
        draft = names.index("draft_work_order") if "draft_work_order" in names else 999
        return inspect < draft
    if scenario.name == "S2":
        draft = names.index("draft_work_order") if "draft_work_order" in names else 999
        approval = names.index("request_human_approval") if "request_human_approval" in names else 999
        return draft < approval
    if scenario.name == "S3":
        return len(run.work_orders) == 1 and run.work_orders[0]["priority"] == "low" and "request_human_approval" not in names
    if scenario.name == "S4":
        return "request_resurvey" in names
    if scenario.name == "S5":
        return not run.work_orders
    return run.tool_call_count <= 25 and names.count("finalize") == 1


def run_evaluation(output_path: Path) -> list[dict[str, object]]:
    root = output_path.parent / ".runs"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        scenario = build_scenario(name)
        run = run_scenario(scenario, root)
        rows.append(
            {
                "scenario": name,
                "pass": evaluate_run(scenario, run),
                "tool_calls": run.tool_call_count,
                "forced_finalize": run.forced_finalize,
                "tools": [entry.tool for entry in run.trace.entries],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _instance(
    instance_id: int,
    class_name: str,
    confidence: float,
    severity: int,
    frame_confs: list[float],
    frame_classes: list[str],
) -> SurveyInstance:
    return SurveyInstance(
        id=instance_id,
        bbox=[450.0, 100.0, 600.0, 300.0],
        class_name=class_name,
        confidence=confidence,
        fused_conf=confidence,
        severity=severity,
        area_m2=0.3,
        lat=28.6139,
        lon=77.2090,
        keyframe_ids=[0],
        frame_confs=frame_confs,
        frame_classes=frame_classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["mock"], default="mock")
    args = parser.parse_args()
    rows = run_evaluation(ROOT / "eval" / "results" / f"agent_{args.llm}.json")
    print("scenario\tpass\ttool_calls")
    for row in rows:
        print(f"{row['scenario']}\t{row['pass']}\t{row['tool_calls']}")


if __name__ == "__main__":
    main()
