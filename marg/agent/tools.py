import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from marg.store.local import LocalStore
from marg.vision.detector import DNNDetector
from marg.vision.models import Detection, SurveyInstance, SurveyResult


TOOL_SPECS: list[dict[str, object]] = [
    {
        "toolSpec": {
            "name": "inspect_roi",
            "description": "Re-detect one uncertain damage instance on an expanded, upscaled best keyframe crop.",
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
            "inputSchema": {"json": {"type": "object", "properties": {"segment_id": {"type": "integer"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]}, "summary": {"type": "string"}}, "required": ["segment_id", "priority", "summary"]}},
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

    def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
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
        return function(**arguments)

    def inspect_roi(self, instance_id: int) -> dict[str, object]:
        instance = self._instance(instance_id)
        if instance is None:
            return {"detections": [], "confirmed": False, "error": "instance not found"}
        source = self._best_keyframe(instance)
        if source is None:
            return {"detections": [], "confirmed": False, "error": "keyframe not found"}
        image = cv2.imread(str(source))
        if image is None:
            return {"detections": [], "confirmed": False, "error": "keyframe unreadable"}
        x, y, width, height = instance.bbox
        expand_width = width * 1.5
        expand_height = height * 1.5
        x0 = max(0, int(round(x + width / 2.0 - expand_width / 2.0)))
        y0 = max(0, int(round(y + height / 2.0 - expand_height / 2.0)))
        x1 = min(image.shape[1], int(round(x + width / 2.0 + expand_width / 2.0)))
        y1 = min(image.shape[0], int(round(y + height / 2.0 + expand_height / 2.0)))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return {"detections": [], "confirmed": False, "error": "empty crop"}
        upscaled = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        crop_dir = self.context.keyframe_dir.parent / "agent_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"inspect_inst_{instance_id:04d}.jpg"
        cv2.imwrite(str(crop_path), upscaled, [cv2.IMWRITE_JPEG_QUALITY, 92])
        detections: list[dict[str, object]] = []
        for detection in self.context.detector.detect(upscaled):
            mapped = [
                float(x0 + detection.bbox[0] / 2.0),
                float(y0 + detection.bbox[1] / 2.0),
                float(detection.bbox[2] / 2.0),
                float(detection.bbox[3] / 2.0),
            ]
            detections.append(
                {"class": detection.class_name, "conf": detection.confidence, "bbox": mapped}
            )
        confirmed = any(
            item["class"] == instance.class_name and float(item["conf"]) >= 0.5
            for item in detections
        )
        return {"detections": detections, "confirmed": confirmed, "crop_path": str(crop_path)}

    def compare_frames(self, instance_id: int) -> dict[str, object]:
        instance = self._instance(instance_id)
        if instance is None:
            return {"frame_confs": [], "classes_seen": [], "agreement_ratio": 0.0}
        classes = instance.frame_classes or [instance.class_name] * len(instance.frame_confs)
        agreement = max(Counter(classes).values(), default=0) / max(1, len(classes))
        return {
            "frame_confs": instance.frame_confs,
            "classes_seen": classes,
            "agreement_ratio": agreement,
        }

    def request_resurvey(self, segment_id: int, reason: str) -> dict[str, object]:
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

    def draft_work_order(self, segment_id: int, priority: str, summary: str) -> dict[str, object]:
        if priority not in {"low", "medium", "high"}:
            return {"error": "priority must be low, medium, or high"}
        work_order_id = uuid.uuid4().hex
        value: dict[str, object] = {
            "work_order_id": work_order_id,
            "segment_id": segment_id,
            "priority": priority,
            "summary": summary,
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

    def _best_keyframe(self, instance: SurveyInstance) -> Path | None:
        for path in self.context.result.keyframe_paths:
            if any(f"kf_{keyframe_id:05d}" in Path(path).name for keyframe_id in instance.keyframe_ids):
                return Path(path)
        for keyframe_id in instance.keyframe_ids:
            path = self.context.keyframe_dir / f"kf_{keyframe_id:05d}.jpg"
            if path.exists():
                return path
        if len(self.context.result.keyframe_paths) == 1:
            path = Path(self.context.result.keyframe_paths[0])
            if path.exists():
                return path
        return None
