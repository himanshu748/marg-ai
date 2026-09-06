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
from .privacy import detect_sensitive_regions, redact
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "keyframes").mkdir(exist_ok=True)
    (output_dir / "crops").mkdir(exist_ok=True)
    (output_dir / "evidence").mkdir(exist_ok=True)
    (output_dir / "evidence_raw").mkdir(exist_ok=True)
    (output_dir / "keyframes_raw").mkdir(exist_ok=True)
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
    evidence_paths: dict[int, str] = {}
    redactions = 0
    stage_times = {"decode": 0.0, "quality": 0.0, "keyframes": 0.0, "detect": 0.0, "track": 0.0}
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
        for update in updates:
            if update.best_observation is not None:
                best_frames[update.instance_id] = packet.frame.copy()
                evidence_path = output_dir / "evidence" / f"inst_{update.instance_id:04d}.jpg"
                raw_path = output_dir / "evidence_raw" / evidence_path.name
                cv2.imwrite(str(raw_path), packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                evidence_frame = packet.frame
                if config.redact:
                    evidence_frame, count = redact(
                        packet.frame,
                        detect_sensitive_regions(packet.frame),
                        protect=[tuple(update.best_observation.bbox)],
                    )
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
            raw_path = output_dir / "keyframes_raw" / path.name
            cv2.imwrite(str(raw_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if config.redact:
                annotated, count = redact(
                    annotated,
                    detect_sensitive_regions(annotated),
                    protect=[tuple(detection.bbox) for detection in detections],
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
) -> int:
    if frame is None:
        return 0
    x, y, width, height = [round(value) for value in bbox]
    crop = frame[max(0, y) : min(frame.shape[0], y + height), max(0, x) : min(frame.shape[1], x + width)]
    if crop.size:
        count = 0
        if redact_output:
            crop, count = redact(crop, detect_sensitive_regions(crop), protect=[(0.0, 0.0, float(crop.shape[1]), float(crop.shape[0]))])
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
