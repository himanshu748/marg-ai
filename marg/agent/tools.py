import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from marg.store.local import LocalStore
from marg.vision.detector import DNNDetector
from marg.vision.models import SurveyInstance, SurveyResult
from marg.vision.privacy import detect_sensitive_regions, redact

TOOL_SPECS: list[dict[str, object]] = [
    {
        "toolSpec": {
            "name": "inspect_roi",
            "description": "Re-detect one uncertain damage instance in a context-aware evidence crop.",
            "inputSchema": {"json": {"type": "object", "properties": {"instance_id": {"type": "integer"}}, "required": ["instance_id"]}},
        }
    },
    {
        "toolSpec": {
            "name": "compare_frames",
            "description": "Compare confidence and class agreement across frames for one instance.",
            "inputSchema": {"json": {"type": "object", "properties": {"instance_id": {"type": "integer"}}, "required": ["instance_id"]}},
        }
    },
    {
        "toolSpec": {
            "name": "request_resurvey",
            "description": "Request a future road re-survey for a segment.",
            "inputSchema": {"json": {"type": "object", "properties": {"segment_id": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["segment_id", "reason"]}},
        }
    },
    {
        "toolSpec": {
            "name": "draft_work_order",
            "description": "Draft a municipal work order for a road segment.",
            "inputSchema": {"json": {"type": "object", "properties": {"segment_id": {"type": "integer"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]}, "summary": {"type": "string"}, "reason": {"type": "string"}}, "required": ["segment_id", "priority", "summary", "reason"]}},
        }
    },
    {
        "toolSpec": {
            "name": "request_human_approval",
            "description": "Request human approval for a drafted work order.",
            "inputSchema": {"json": {"type": "object", "properties": {"work_order_id": {"type": "string"}}, "required": ["work_order_id"]}},
        }
    },
    {
        "toolSpec": {
            "name": "dismiss_instance",
            "description": "Dismiss an unconfirmed or otherwise unreliable instance.",
            "inputSchema": {"json": {"type": "object", "properties": {"instance_id": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["instance_id", "reason"]}},
        }
    },
    {
        "toolSpec": {
            "name": "finalize",
            "description": "Finalize the agent review with a concise honest summary.",
            "inputSchema": {"json": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
        }
    },
]


@dataclass(slots=True)
class ToolContext:
    result: SurveyResult
    store: LocalStore
    detector: DNNDetector
    keyframe_dir: Path
    work_orders: list[dict[str, object]] = field(default_factory=list)
    resurveys: list[dict[str, object]] = field(default_factory=list)
    dismissals: list[dict[str, object]] = field(default_factory=list)


class ToolSet:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self.finalized = False
        self.summary = ""
        self.inspections: dict[int, dict[str, object]] = {}

    def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if self.finalized:
            return {"error": "Review has already been finalized"}
        functions = {
            "inspect_roi": self.inspect_roi,
            "compare_frames": self.compare_frames,
            "request_resurvey": self.request_resurvey,
            "draft_work_order": self.draft_work_order,
            "request_human_approval": self.request_human_approval,
            "dismiss_instance": self.dismiss_instance,
            "finalize": self.finalize,
        }
        function = functions.get(name)
        if function is None:
            return {"error": f"Unknown tool: {name}"}
        spec = next(item["toolSpec"]["inputSchema"]["json"] for item in TOOL_SPECS if item["toolSpec"]["name"] == name)
        properties = spec["properties"]
        if any(key not in properties for key in arguments) or any(key not in arguments for key in spec["required"]):
            return {"error": "Tool arguments do not match the declared schema"}
        for key, value in arguments.items():
            expected = properties[key]
            if expected["type"] == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                return {"error": f"{key} must be an integer"}
            if expected["type"] == "string" and (not isinstance(value, str) or not value.strip() or len(value) > 4000):
                return {"error": f"{key} must be a nonempty string of at most 4000 characters"}
            if "enum" in expected and value not in expected["enum"]:
                return {"error": f"Invalid {key}"}
        output = function(**arguments)
        if name == "inspect_roi":
            self.inspections[int(arguments["instance_id"])] = output
        return output

    def inspect_roi(self, instance_id: int) -> dict[str, object]:
        instance = self._instance(instance_id)
        if instance is None:
            return {"detections": [], "confirmed": False, "error": "instance not found"}
        source = self._evidence_path(instance) or self._best_keyframe(instance)
        if source is None:
            return {"detections": [], "confirmed": False, "error": "keyframe not found"}
        image = cv2.imread(str(source))
        if image is None:
            return {"detections": [], "confirmed": False, "error": "keyframe unreadable"}
        observation = max(
            instance.observations,
            key=lambda item: item.fused_conf,
            default=None,
        )
        bbox = observation.bbox if observation is not None else instance.bbox
        x, y, width, height = bbox
        side = max(width * 3.0, height * 3.0, 320.0)
        center_x, center_y = x + width / 2.0, y + height / 2.0
        x0 = max(0, min(image.shape[1] - 1, round(center_x - side / 2.0)))
        y0 = max(0, min(image.shape[0] - 1, round(center_y - side / 2.0)))
        x1 = min(image.shape[1], max(x0 + 1, round(center_x + side / 2.0)))
        y1 = min(image.shape[0], max(y0 + 1, round(center_y + side / 2.0)))
        x0 = max(0, x1 - round(side)) if x1 == image.shape[1] else x0
        y0 = max(0, y1 - round(side)) if y1 == image.shape[0] else y0
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return {"detections": [], "confirmed": False, "error": "empty crop"}
        scale = 640.0 / max(crop.shape[:2])
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interpolation)
        crop_dir = self.context.keyframe_dir.parent / "agent_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"inspect_inst_{instance_id:04d}.jpg"
        detections: list[dict[str, object]] = []
        mapped_detections: list[tuple[dict[str, object], float]] = []
        for detection in self.context.detector.detect(resized):
            mapped = [
                float(x0 + detection.bbox[0] / scale),
                float(y0 + detection.bbox[1] / scale),
                float(detection.bbox[2] / scale),
                float(detection.bbox[3] / scale),
            ]
            item = {"class": detection.class_name, "conf": detection.confidence, "bbox": mapped}
            detections.append(item)
            mapped_detections.append((item, _bbox_iou(tuple(bbox), tuple(mapped))))
        annotated = resized.copy()
        for item, _ in mapped_detections:
            mapped = item["bbox"]
            dx, dy, dw, dh = [round((float(value) - (x0 if index % 2 == 0 else y0)) * scale) for index, value in enumerate(mapped)]
            cv2.rectangle(annotated, (dx, dy), (dx + round(dw), dy + round(dh)), (0, 220, 0), 2)
            cv2.putText(annotated, f"{item['class']} {float(item['conf']):.2f}", (dx, max(16, dy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        original = [
            round((value - (x0 if index % 2 == 0 else y0)) * scale)
            for index, value in enumerate(bbox)
        ]
        cv2.rectangle(
            annotated,
            (original[0], original[1]),
            (original[0] + original[2], original[1] + original[3]),
            (0, 0, 255),
            2,
        )
        protect = [(float(original[0]), float(original[1]), float(original[2]), float(original[3]))]
        annotated, _ = redact(annotated, detect_sensitive_regions(annotated), protect=protect)
        cv2.imwrite(str(crop_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        matching = [(item, overlap) for item, overlap in mapped_detections if item["class"] == instance.class_name]
        best_match, best_iou = max(
            matching,
            key=lambda pair: float(pair[0]["conf"]),
            default=({"conf": 0.0}, 0.0),
        )
        confirmed = any(
            (
                float(item["conf"]) >= 0.5
                or float(item["conf"]) >= instance.confidence + 0.10
            )
            and overlap >= 0.3
            for item, overlap in matching
        )
        return {
            "detections": detections,
            "confirmed": confirmed,
            "best_match_conf": float(best_match.get("conf", 0.0)),
            "iou": float(best_iou),
            "crop_path": crop_path.name,
        }

    def compare_frames(self, instance_id: int) -> dict[str, object]:
        instance = self._instance(instance_id)
        if instance is None:
            return {"frame_confs": [], "classes_seen": [], "agreement_ratio": 0.0}
        frame_confs = [observation.conf for observation in instance.observations]
        classes = [observation.class_name for observation in instance.observations]
        frame_confs = frame_confs or instance.frame_confs
        classes = classes or instance.frame_classes or [instance.class_name] * len(frame_confs)
        agreement = max(Counter(classes).values(), default=0) / max(1, len(classes))
        return {
            "frame_confs": frame_confs,
            "classes_seen": classes,
            "agreement_ratio": agreement,
        }

    def request_resurvey(self, segment_id: int, reason: str) -> dict[str, object]:
        if not any(item.id == segment_id for item in self.context.result.segments):
            return {"error": "segment not found"}
        existing = next((item for item in self.context.resurveys if item["segment_id"] == segment_id), None)
        if existing is not None:
            return existing
        resurvey_id = uuid.uuid4().hex
        value: dict[str, object] = {
            "resurvey_id": resurvey_id,
            "segment_id": segment_id,
            "reason": reason,
            "status": "requested",
        }
        self.context.resurveys.append(value)
        self.context.store.save_resurvey(self.context.result.survey_id, resurvey_id, value)
        return value

    def draft_work_order(
        self,
        segment_id: int,
        priority: str,
        summary: str,
        reason: str = "",
    ) -> dict[str, object]:
        if priority not in {"low", "medium", "high"}:
            return {"error": "priority must be low, medium, or high"}
        segment = next((item for item in self.context.result.segments if item.id == segment_id), None)
        if segment is None:
            return {"error": "segment not found"}
        dismissed = {item["instance_id"] for item in self.context.dismissals}
        instance_ids = [item for item in segment.instance_ids if item not in dismissed]
        instances = [
            item for item in self.context.result.instances if item.id in instance_ids
        ]
        if not instances:
            return {"error": "segment has no actionable damage instances"}
        if any(item.fused_conf < 0.55 and not self.inspections.get(item.id, {}).get("confirmed") for item in instances):
            return {"error": "low-confidence evidence must be confirmed by inspection before drafting"}
        required_priority = "high" if any(item.severity == 5 for item in instances) else "medium"
        needs_approval = any(item.severity >= 4 for item in instances) or sum(item.class_name == "D40" for item in instances) >= 3
        if needs_approval and priority != required_priority:
            return {"error": f"this segment requires {required_priority} priority and human approval"}
        if not needs_approval and (priority != "low" or len(instances) < 2 or any(item.severity > 2 for item in instances)):
            return {"error": "this segment does not meet the work-order policy; retain the evidence or request a resurvey"}
        existing = next((item for item in self.context.work_orders if item["segment_id"] == segment_id), None)
        if existing is not None:
            return existing
        instance_ids = [item.id for item in instances]
        total_area = sum(item.area_m2 for item in instances)
        max_severity = max((item.severity for item in instances), default=0)
        class_counts = Counter(item.class_name for item in instances)
        class_text = ", ".join(
            f"{count} {class_name}" for class_name, count in sorted(class_counts.items())
        )
        is_potholes = set(class_counts) == {"D40"}
        noun = (
            "pothole" if len(instances) == 1 else "potholes"
        ) if is_potholes else "road defects"
        title = (
            f"Pothole repair — {class_text} {noun}, "
            f"~{round(total_area)} m², sev {max_severity}"
            if is_potholes
            else f"Road repair — {class_text} {noun}, ~{round(total_area)} m², sev {max_severity}"
        )
        instance_details = [
            {
                "id": item.id,
                "class": item.class_name,
                "severity": item.severity,
                "area_m2": item.area_m2,
                "fused_conf": item.fused_conf,
                "lat": item.lat,
                "lon": item.lon,
                "evidence": Path(item.evidence_path).name if item.evidence_path else None,
            }
            for item in instances
        ]
        work_order_id = uuid.uuid4().hex
        value: dict[str, object] = {
            "work_order_id": work_order_id,
            "segment_id": segment_id,
            "instance_ids": instance_ids,
            "instances": instance_details,
            "priority": priority,
            "title": title,
            "summary": summary,
            "reason": reason,
            "status": "draft",
            "approval_requested": False,
        }
        self.context.work_orders.append(value)
        self.context.store.save_work_order(self.context.result.survey_id, work_order_id, value)
        return value

    def request_human_approval(self, work_order_id: str) -> dict[str, object]:
        for work_order in self.context.work_orders:
            if work_order.get("work_order_id") == work_order_id:
                work_order["approval_requested"] = True
                work_order["status"] = "pending_approval"
                self.context.store.save_work_order(
                    self.context.result.survey_id, work_order_id, work_order
                )
                return {"work_order_id": work_order_id, "status": "pending_approval"}
        return {"work_order_id": work_order_id, "status": "not_found"}

    def dismiss_instance(self, instance_id: int, reason: str) -> dict[str, object]:
        if self._instance(instance_id) is None:
            return {"error": "instance not found"}
        if any(instance_id in item.get("instance_ids", []) for item in self.context.work_orders):
            return {"error": "instance already supports a work order; rerun the review to revise its evidence"}
        existing = next((item for item in self.context.dismissals if item["instance_id"] == instance_id), None)
        if existing is not None:
            return existing
        value: dict[str, object] = {
            "instance_id": instance_id,
            "reason": reason,
            "status": "dismissed",
        }
        self.context.dismissals.append(value)
        self.context.store.save_dismissal(self.context.result.survey_id, instance_id, value)
        return value

    def finalize(self, summary: str) -> dict[str, object]:
        self.finalized = True
        self.summary = summary
        return {"status": "finalized", "summary": summary}

    def _instance(self, instance_id: int) -> SurveyInstance | None:
        return next((item for item in self.context.result.instances if item.id == instance_id), None)

    def _evidence_path(self, instance: SurveyInstance) -> Path | None:
        if instance.evidence_path is None:
            return None
        path = Path(instance.evidence_path)
        if not path.is_absolute():
            path = self.context.keyframe_dir.parent / path
        raw_path = path.parent.parent / "evidence_raw" / path.name
        if raw_path.is_file():
            return raw_path
        return path if path.is_file() else None

    def _best_keyframe(self, instance: SurveyInstance) -> Path | None:
        for path in self.context.result.keyframe_paths:
            resolved = self._resolve_path(path)
            if any(f"kf_{keyframe_id:05d}" in resolved.name for keyframe_id in instance.keyframe_ids):
                return resolved
        for keyframe_id in instance.keyframe_ids:
            path = self.context.keyframe_dir / f"kf_{keyframe_id:05d}.jpg"
            if path.exists():
                return path
        for candidate in self.context.result.keyframe_paths:
            path = self._resolve_path(candidate)
            if path.exists():
                return path
        return None

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.context.keyframe_dir.parent / path


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0
