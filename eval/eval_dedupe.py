import argparse
import json
from pathlib import Path

MANUAL_VISIBLE_POTHOLES = {
    "pothole_cars": 1,
    "pothole_kumasi": 2,
}

BEFORE = {
    "pothole_cars": {
        "detections": 21,
        "instances": 4,
        "observation_counts": [2, 1, 4, 14],
    },
    "pothole_kumasi": {
        "detections": 96,
        "instances": 3,
        "observation_counts": [62, 32, 2],
    },
}


def evaluate_survey(path: Path) -> dict[str, object]:
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    detections = sum(int(value) for value in result.get("detections", {}).values())
    instances = result.get("instances", [])
    observations = [
        len(instance.get("observations", []))
        for instance in instances
        if isinstance(instance, dict)
    ]
    visible = MANUAL_VISIBLE_POTHOLES[path.name]
    return {
        "survey": path.name,
        "detections": detections,
        "instances": len(instances),
        "compression_ratio": detections / max(1, len(instances)),
        "observation_counts": observations,
        "hand_count_visible_potholes": visible,
        "unique_instance_precision": visible / max(1, len(instances)),
        "unique_instance_recall": min(1.0, len(instances) / max(1, visible)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/dedupe.md"))
    args = parser.parse_args()
    rows = [
        evaluate_survey(args.outputs / "pothole_cars"),
        evaluate_survey(args.outputs / "pothole_kumasi"),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# Detections-to-instance deduplication",
        "",
        (
            "Manual ground truth is a visual count of distinct pothole regions across "
            "the evidence frames and keyframes, not a frame-by-frame count. "
            "Occluded or partially visible regions were counted once."
        ),
        "",
        "## Before observation gating",
        "",
        "| Survey | Detections | Instances | Compression | Visible potholes | Unique precision | Unique recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for survey, values in BEFORE.items():
        visible = MANUAL_VISIBLE_POTHOLES[survey]
        instances = int(values["instances"])
        detections = int(values["detections"])
        markdown.append(
            f"| {survey} | {detections} | {instances} | "
            f"{detections / max(1, instances):.2f}:1 | {visible} | "
            f"{visible / max(1, instances):.3f} | "
            f"{min(1.0, instances / max(1, visible)):.3f} |"
        )
    markdown.extend(
        [
            "",
            "| Survey | Per-instance observation counts |",
            "| --- | --- |",
        ]
    )
    for survey, values in BEFORE.items():
        markdown.append(f"| {survey} | {', '.join(str(value) for value in values['observation_counts'])} |")
    markdown.extend(
        [
            "",
            "## After observation gating (`min_observations=2`)",
            "",
            "| Survey | Detections | Instances | Compression | Visible potholes | Unique precision | Unique recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        markdown.append(
            f"| {row['survey']} | {row['detections']} | {row['instances']} | "
            f"{row['compression_ratio']:.2f}:1 | {row['hand_count_visible_potholes']} | "
            f"{row['unique_instance_precision']:.3f} | "
            f"{row['unique_instance_recall']:.3f} |"
        )
    markdown.extend(["", "| Survey | Per-instance observation counts |", "| --- | --- |"])
    for row in rows:
        markdown.append(
            f"| {row['survey']} | {', '.join(str(value) for value in row['observation_counts'])} |"
        )
    args.out.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
