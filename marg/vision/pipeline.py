import argparse
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .classical import fuse_detection
from .config import VisionConfig
from .detector import DNNDetector
from .geo import GeoTrack, cluster_instances
from .ingest import VideoSource
from .keyframes import KeyframeSelector
from .models import Detection, SurveyInstance, SurveyResult, SurveySummary
from .privacy import detect_sensitive_regions, ensure_privacy_ready, redact
from .quality import assess
from .severity import bbox_area_m2, score
from .tracker import InstanceTracker


def run(
    video: str | Path,
    gpx: str | Path | None = None,
    out_dir: str | Path = "out",
    config: VisionConfig | None = None,
) -> SurveyResult:
    started = time.perf_counter()
    video_path = Path(video)
    output_dir = Path(out_dir)
    config = config or VisionConfig()
    if config.redact:
        ensure_privacy_ready()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A restarted run must not expose an old completion record with new media.
    (output_dir / "result.json").unlink(missing_ok=True)
    (output_dir / "keyframes").mkdir(exist_ok=True)
    (output_dir / "crops").mkdir(exist_ok=True)
    (output_dir / "evidence").mkdir(exist_ok=True)
    (output_dir / "evidence_raw").mkdir(exist_ok=True)
    (output_dir / "agent_crops").mkdir(exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    for media_dir in ("evidence", "evidence_raw", "crops", "keyframes", "agent_crops", "samples"):
        for media_path in (output_dir / media_dir).iterdir():
            if media_path.is_file() or media_path.is_symlink():
                media_path.unlink()
    model_path = config.model_file(Path(__file__).resolve().parents[2])
    detector = DNNDetector(model_path, config)
    selector = KeyframeSelector(config)
    tracker = InstanceTracker(config)
    geo = GeoTrack.from_path(gpx, config)
    frame_count = 0
    processed_frames = 0
    keyframe_paths: list[str] = []
    crop_paths: list[str] = []
    detections_by_class = {class_name: 0 for class_name in config.classes}
    previous_gray: np.ndarray | None = None
    best_frames: dict[int, np.ndarray] = {}
    best_sensitive_regions: dict[int, list[tuple[int, int, int, int]]] = {}
    evidence_paths: dict[int, str] = {}
    redactions = 0
    stage_times = {
        "decode": 0.0,
        "quality": 0.0,
        "keyframes": 0.0,
        "detect": 0.0,
        "track": 0.0,
        "privacy_evidence": 0.0,
    }
    evidence_redaction_frames = 0
    keyframe_id_for_frame: int | None = None
    example_keyframes: list[tuple[int, np.ndarray, list[tuple[int, Detection]]]] = []
    source = VideoSource(video_path, config)
    for packet in source:
        stage_times["decode"] += source.last_decode_s
        frame_count = packet.index + 1
        if packet.index % config.frame_stride != 0:
            continue
        processed_frames += 1
        quality_started = time.perf_counter()
        quality = assess(packet.frame, config)
        stage_times["quality"] += time.perf_counter() - quality_started
        if not quality.accepted:
            continue
        keyframe_started = time.perf_counter()
        keyframe = selector.consider(packet.frame, packet.index, packet.timestamp_s)
        stage_times["keyframes"] += time.perf_counter() - keyframe_started
        keyframe_id_for_frame = keyframe.keyframe_id if keyframe is not None else None
        current_gray = _flow_gray(packet.frame, config.optical_flow_width)
        flow = None
        if previous_gray is not None:
            flow_small = cv2.calcOpticalFlowFarneback(
                previous_gray,
                current_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            flow = cv2.resize(
                flow_small,
                (packet.frame.shape[1], packet.frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            flow *= np.float32(
                [packet.frame.shape[1] / max(1, flow_small.shape[1]), packet.frame.shape[0] / max(1, flow_small.shape[0])]
            )
        detect_started = time.perf_counter()
        detections = [fuse_detection(packet.frame, detection) for detection in detector.detect(packet.frame)]
        stage_times["detect"] += time.perf_counter() - detect_started
        for detection in detections:
            detections_by_class[detection.class_name] += 1
        lat, lon = geo.sample(packet.timestamp_s)
        track_started = time.perf_counter()
        updates = tracker.update(
            detections,
            keyframe_id_for_frame,
            flow,
            lat,
            lon,
            frame_idx=packet.index,
            t_s=packet.timestamp_s,
        )
        stage_times["track"] += time.perf_counter() - track_started
        sensitive_regions = None
        if config.redact and (keyframe is not None or any(update.best_observation is not None for update in updates)):
            # Detect on the unannotated frame; overlays can obscure faces/plates.
            redact_started = time.perf_counter()
            sensitive_regions = detect_sensitive_regions(packet.frame)
            stage_times["privacy_evidence"] += time.perf_counter() - redact_started
        for update in updates:
            if update.best_observation is not None:
                best_frames[update.instance_id] = packet.frame.copy()
                evidence_path = output_dir / "evidence" / f"inst_{update.instance_id:04d}.jpg"
                raw_path = output_dir / "evidence_raw" / evidence_path.name
                cv2.imwrite(str(raw_path), packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                evidence_frame = packet.frame
                if config.redact:
                    redact_started = time.perf_counter()
                    evidence_frame, count = redact(
                        packet.frame,
                        sensitive_regions or [],
                    )
                    best_sensitive_regions[update.instance_id] = sensitive_regions or []
                    stage_times["privacy_evidence"] += time.perf_counter() - redact_started
                    evidence_redaction_frames += 1
                    redactions += count
                cv2.imwrite(str(evidence_path), evidence_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                evidence_paths[update.instance_id] = str(evidence_path.relative_to(output_dir))
        if keyframe is not None:
            annotated = keyframe.frame.copy()
            update_by_detection = {id(update.detection): update.instance_id for update in updates}
            rendered: list[tuple[int, Detection]] = []
            for detection in detections:
                instance_id = update_by_detection.get(id(detection), -1)
                _draw_detection(annotated, detection, instance_id)
                rendered.append((instance_id, detection))
            path = output_dir / "keyframes" / f"kf_{keyframe.keyframe_id:05d}.jpg"
            if config.redact:
                annotated, count = redact(
                    annotated,
                    sensitive_regions or [],
                )
                redactions += count
            cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            keyframe_paths.append(str(path.relative_to(output_dir)))
            if len(example_keyframes) < 6:
                example_keyframes.append((keyframe.keyframe_id, annotated, rendered))
        previous_gray = current_gray
    instances: list[SurveyInstance] = []
    for track in tracker.finalize():
        class_name, confidence, frame_confs = tracker.summarize(track)
        lat, lon = track.lat, track.lon
        best_observation = track.best_observation
        best_bbox = (
            tuple(best_observation.bbox)
            if best_observation is not None
            else track.bbox
        )
        area = bbox_area_m2(best_bbox, config.target_width, int(config.target_width * 9 / 16), config)
        instance = SurveyInstance(
            id=track.instance_id,
            bbox=[float(value) for value in best_bbox],
            class_name=class_name,
            confidence=confidence,
            fused_conf=track.fused_conf,
            severity=score(class_name, area),
            area_m2=area,
            lat=lat,
            lon=lon,
            keyframe_ids=track.keyframe_ids,
            frame_confs=frame_confs,
            frame_classes=tracker.classes(track),
            observations=track.observations,
            best_frame_idx=best_observation.frame_idx if best_observation is not None else -1,
            evidence_path=evidence_paths.get(track.instance_id),
        )
        instances.append(instance)
        crop_path = output_dir / "crops" / f"inst_{track.instance_id:04d}.jpg"
        if config.redact:
            redactions += _save_crop(
                best_frames.get(track.instance_id),
                best_bbox,
                crop_path,
                redact_output=True,
                sensitive_regions=best_sensitive_regions.get(track.instance_id),
            )
        else:
            _save_crop(best_frames.get(track.instance_id), best_bbox, crop_path)
        if crop_path.exists():
            crop_paths.append(str(crop_path.relative_to(output_dir)))
    segments = cluster_instances(instances, config.geo_cluster_radius_m)
    metrics = {
        "runtime_s": time.perf_counter() - started,
        "fps_processed": processed_frames / max(0.001, time.perf_counter() - started),
        "redactions": redactions,
        "evidence_redaction_frames": float(evidence_redaction_frames),
        "evidence_redaction_avg_ms": (
            stage_times["privacy_evidence"] * 1000 / evidence_redaction_frames
            if evidence_redaction_frames
            else 0.0
        ),
        **{f"stage_{name}_s": value for name, value in stage_times.items()},
    }
    result = SurveyResult(
        survey_id=uuid4().hex,
        video=str(video_path),
        gpx=str(gpx) if gpx is not None else None,
        geo_source=geo.source,
        frames=frame_count,
        processed_frames=processed_frames,
        keyframes=len(keyframe_paths),
        detections=detections_by_class,
        surveys=[],
        instances=instances,
        segments=segments,
        keyframe_paths=keyframe_paths,
        crop_paths=crop_paths,
        metrics=metrics,
        redactions=redactions,
    )
    result.surveys = [
        SurveySummary(id=result.survey_id, video=str(video_path), geo_source=geo.source)
    ]
    (output_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    _save_examples(example_keyframes, output_dir)
    return result


def _flow_gray(frame: np.ndarray, width: int) -> np.ndarray:
    scale = width / frame.shape[1]
    height = max(1, int(frame.shape[0] * scale))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _draw_detection(frame: np.ndarray, detection: Detection, instance_id: int) -> None:
    x, y, width, height = [round(value) for value in detection.bbox]
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
    label = f"{instance_id}:{detection.class_name} {detection.fused_conf:.2f}"
    cv2.putText(frame, label, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)


def _save_crop(
    frame: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    path: Path,
    redact_output: bool = False,
    sensitive_regions: list[tuple[int, int, int, int]] | None = None,
) -> int:
    if frame is None:
        return 0
    x, y, width, height = [round(value) for value in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = frame[y0:y1, x0:x1]
    if crop.size:
        count = 0
        if redact_output:
            regions = sensitive_regions if sensitive_regions is not None else detect_sensitive_regions(frame)
            crop_regions = []
            for sx, sy, sw, sh in regions:
                left, top, right, bottom = max(x0, sx), max(y0, sy), min(x1, sx + sw), min(y1, sy + sh)
                if right > left and bottom > top:
                    crop_regions.append((left - x0, top - y0, right - left, bottom - top))
            crop, count = redact(crop, crop_regions)
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return count
    return 0


def _save_examples(
    examples: list[tuple[int, np.ndarray, list[tuple[int, Detection]]]],
    output_dir: Path,
) -> None:
    samples = output_dir / "samples"
    samples.mkdir(exist_ok=True)
    for keyframe_id, frame, _ in examples:
        cv2.imwrite(str(samples / f"kf_{keyframe_id:05d}_annotated.jpg"), frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MargAI OpenCV road-damage pipeline.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--gpx")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    result = run(args.video, args.gpx, args.out)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
