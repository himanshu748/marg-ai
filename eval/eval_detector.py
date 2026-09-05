import argparse
import json
import platform
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector

CLASSES = ("D00", "D10", "D20", "D40")


@dataclass(frozen=True, slots=True)
class Box:
    class_name: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Prediction:
    image_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    path: Path
    ground_truth: list[Box]


def intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute all-point interpolated average precision."""
    if recall.size == 0 or precision.size == 0:
        return 0.0
    order = np.argsort(recall)
    recalls = np.asarray(recall, dtype=np.float64)[order]
    precisions = np.asarray(precision, dtype=np.float64)[order]
    envelope = np.maximum.accumulate(precisions[::-1])[::-1]
    points = np.concatenate(([0.0], recalls, [1.0]))
    values = np.concatenate(([envelope[0]], envelope, [0.0]))
    return float(np.sum((points[1:] - points[:-1]) * values[1:]))


def parse_xml(path: Path, image_width: int, image_height: int) -> list[Box]:
    root = ET.parse(path).getroot()
    boxes: list[Box] = []
    for object_node in root.findall("object"):
        name = object_node.findtext("name", "")
        box = object_node.find("bndbox")
        if name not in CLASSES or box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        boxes.append(
            Box(
                name,
                (
                    max(0.0, xmin),
                    max(0.0, ymin),
                    max(1.0, min(float(image_width), xmax) - xmin),
                    max(1.0, min(float(image_height), ymax) - ymin),
                ),
            )
        )
    return boxes


def parse_yolo(path: Path, image_width: int, image_height: int) -> list[Box]:
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id, center_x, center_y, width, height = (
            int(fields[0]),
            float(fields[1]),
            float(fields[2]),
            float(fields[3]),
            float(fields[4]),
        )
        if class_id < 0 or class_id >= len(CLASSES):
            continue
        box_width = width * image_width
        box_height = height * image_height
        boxes.append(
            Box(
                CLASSES[class_id],
                (
                    (center_x - width / 2.0) * image_width,
                    (center_y - height / 2.0) * image_height,
                    box_width,
                    box_height,
                ),
            )
        )
    return boxes


def load_records(images_dir: Path, labels_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        relative = image_path.relative_to(images_dir)
        stem = image_path.stem
        xml_path = labels_dir / relative.with_suffix(".xml")
        txt_path = labels_dir / relative.with_suffix(".txt")
        if not xml_path.is_file():
            xml_candidates = list(labels_dir.rglob(f"{stem}.xml"))
            xml_path = xml_candidates[0] if xml_candidates else xml_path
        if not txt_path.is_file():
            txt_candidates = list(labels_dir.rglob(f"{stem}.txt"))
            txt_path = txt_candidates[0] if txt_candidates else txt_path
        if xml_path.is_file():
            ground_truth = parse_xml(xml_path, image.shape[1], image.shape[0])
        elif txt_path.is_file():
            ground_truth = parse_yolo(txt_path, image.shape[1], image.shape[0])
        else:
            ground_truth = []
        records.append(ImageRecord(stem, image_path, ground_truth))
    return records


def _match_predictions(
    predictions: list[Prediction],
    records: list[ImageRecord],
    class_name: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    truth = {
        record.image_id: [
            box for box in record.ground_truth if box.class_name == class_name
        ]
        for record in records
    }
    matched = {image_id: [False] * len(boxes) for image_id, boxes in truth.items()}
    ordered = sorted(
        (prediction for prediction in predictions if prediction.class_name == class_name),
        key=lambda prediction: prediction.confidence,
        reverse=True,
    )
    true_positive: list[int] = []
    false_positive: list[int] = []
    for prediction in ordered:
        candidates = truth.get(prediction.image_id, [])
        overlaps = [
            intersection_over_union(prediction.bbox, box.bbox) if not used else -1.0
            for box, used in zip(candidates, matched.get(prediction.image_id, []))
        ]
        best_index = int(np.argmax(overlaps)) if overlaps else -1
        if best_index >= 0 and overlaps[best_index] >= threshold:
            matched[prediction.image_id][best_index] = True
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)
    total_truth = sum(len(boxes) for boxes in truth.values())
    if not true_positive:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), total_truth
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    return (
        cumulative_tp / np.maximum(1, cumulative_tp + cumulative_fp),
        cumulative_tp / max(1, total_truth),
        total_truth,
    )


def _production_counts(
    predictions: list[Prediction],
    records: list[ImageRecord],
    threshold: float,
) -> tuple[dict[str, dict[str, int]], dict[str, list[tuple[Box, bool]]]]:
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASSES}
    matches: dict[str, list[tuple[Box, bool]]] = {}
    by_image = {record.image_id: record for record in records}
    for image_id, record in by_image.items():
        matches[image_id] = [(box, False) for box in record.ground_truth]
    for class_name in CLASSES:
        image_predictions = [
            prediction
            for prediction in predictions
            if prediction.class_name == class_name
            and prediction.confidence >= threshold
        ]
        for prediction in sorted(
            image_predictions,
            key=lambda item: item.confidence,
            reverse=True,
        ):
            candidates = [
                (index, box)
                for index, (box, matched) in enumerate(matches[prediction.image_id])
                if not matched and box.class_name == class_name
            ]
            overlaps = [
                (index, intersection_over_union(prediction.bbox, box.bbox))
                for index, box in candidates
            ]
            if overlaps:
                best_index, best_overlap = max(overlaps, key=lambda item: item[1])
            else:
                best_index, best_overlap = -1, 0.0
            if best_overlap >= threshold:
                matches[prediction.image_id][best_index] = (
                    matches[prediction.image_id][best_index][0],
                    True,
                )
                counts[class_name]["tp"] += 1
            else:
                counts[class_name]["fp"] += 1
        counts[class_name]["fn"] = sum(
            1
            for record in records
            for box, matched in matches[record.image_id]
            if box.class_name == class_name and not matched
        )
    return counts, matches


def _annotate_failure(
    record: ImageRecord,
    predictions: list[Prediction],
    path: Path,
) -> None:
    image = cv2.imread(str(record.path))
    if image is None:
        return
    for box in record.ground_truth:
        x, y, width, height = [round(value) for value in box.bbox]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 220, 0), 2)
        cv2.putText(image, f"GT {box.class_name}", (x, max(16, y - 5)), 0, 0.5, (0, 220, 0), 1)
    for prediction in predictions:
        x, y, width, height = [round(value) for value in prediction.bbox]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"P {prediction.class_name} {prediction.confidence:.2f}",
            (x, min(image.shape[0] - 4, y + height + 15)),
            0,
            0.5,
            (0, 0, 255),
            1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def evaluate(images_dir: Path, labels_dir: Path, out_path: Path) -> dict[str, object]:
    records = load_records(images_dir, labels_dir)
    config = VisionConfig(conf_thresh=0.001)
    detector = DNNDetector(config.model_file(Path(__file__).resolve().parents[1]), config)
    predictions: list[Prediction] = []
    latencies: list[float] = []
    for record in records:
        image = cv2.imread(str(record.path))
        if image is None:
            continue
        started = time.perf_counter()
        detections = detector.detect(image)
        latencies.append((time.perf_counter() - started) * 1000.0)
        predictions.extend(
            Prediction(record.image_id, item.class_name, item.bbox, item.confidence)
            for item in detections
        )
    ap_by_class: dict[str, float] = {}
    for class_name in CLASSES:
        precision, recall, total_truth = _match_predictions(
            predictions,
            records,
            class_name,
            0.5,
        )
        ap_by_class[class_name] = compute_ap(recall, precision) if total_truth else 0.0
    counts, matches = _production_counts(predictions, records, 0.25)
    total_tp = sum(item["tp"] for item in counts.values())
    total_fp = sum(item["fp"] for item in counts.values())
    total_fn = sum(item["fn"] for item in counts.values())
    precision_at_threshold = total_tp / max(1, total_tp + total_fp)
    recall_at_threshold = total_tp / max(1, total_tp + total_fn)
    failure_candidates: list[tuple[float, str, ImageRecord]] = []
    by_image = {record.image_id: record for record in records}
    for image_id, values in matches.items():
        unmatched = [box for box, matched in values if not matched]
        if unmatched:
            failure_candidates.append(
                (len(unmatched), "fn", by_image[image_id])
            )
        image_predictions = [item for item in predictions if item.image_id == image_id]
        if image_predictions and not any(matched for _, matched in values):
            failure_candidates.append(
                (
                    min(item.confidence for item in image_predictions),
                    "fp",
                    by_image[image_id],
                )
            )
    selected: list[ImageRecord] = []
    for _, _, record in sorted(failure_candidates, reverse=True, key=lambda item: item[0]):
        if record not in selected:
            selected.append(record)
        if len(selected) >= 8:
            break
    failure_dir = Path("docs/failures")
    for index, record in enumerate(selected):
        _annotate_failure(
            record,
            [item for item in predictions if item.image_id == record.image_id],
            failure_dir / f"failure_{index + 1:02d}_{record.image_id}.jpg",
        )
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu_model = platform.processor() or "unknown"
    latency_array = np.asarray(latencies, dtype=np.float64)
    result: dict[str, object] = {
        "source": "TamAko783/Unified_Road_Defect_Dataset RDD-2022 India subset",
        "images": len(records),
        "ground_truth_boxes": sum(len(record.ground_truth) for record in records),
        "predictions": len(predictions),
        "iou_threshold": 0.5,
        "ap_method": "all-point interpolated AP over confidence-ranked detections",
        "ap50": ap_by_class,
        "map50": float(np.mean(list(ap_by_class.values()))) if ap_by_class else 0.0,
        "production_threshold": 0.25,
        "precision": precision_at_threshold,
        "recall": recall_at_threshold,
        "counts": counts,
        "latency_ms_median": float(np.median(latency_array)) if latencies else 0.0,
        "latency_ms_p95": float(np.percentile(latency_array, 95)) if latencies else 0.0,
        "images_per_second": 1000.0 / float(np.mean(latency_array)) if latencies else 0.0,
        "cpu_model": cpu_model,
        "failure_images": [record.image_id for record in selected],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = [
        "# RDD2022 India detector evaluation",
        "",
        f"- Images: **{result['images']}**; ground-truth boxes: **{result['ground_truth_boxes']}**",
        f"- AP: **{result['ap_method']}**",
        f"- Production threshold: **{result['production_threshold']}**; IoU threshold: **0.50**",
        f"- CPU: **{cpu_model}**",
        "",
        "| Class | AP@0.5 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for class_name in CLASSES:
        item = counts[class_name]
        markdown.append(
            f"| {class_name} | {ap_by_class[class_name]:.4f} | "
            f"{item['tp']} | {item['fp']} | {item['fn']} |"
        )
    markdown.extend(
        [
            f"| **mAP** | **{result['map50']:.4f}** | | | |",
            "",
            "| Production metric | Value |",
            "| --- | ---: |",
            f"| Precision @ 0.25 | {precision_at_threshold:.4f} |",
            f"| Recall @ 0.25 | {recall_at_threshold:.4f} |",
            f"| Median inference latency (ms) | {result['latency_ms_median']:.2f} |",
            f"| P95 inference latency (ms) | {result['latency_ms_p95']:.2f} |",
            f"| Inference throughput (images/s) | {result['images_per_second']:.2f} |",
            "",
            "Failure images are in `docs/failures/`; predictions are red and ground truth is green.",
        ]
    )
    out_path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.images, args.labels, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
