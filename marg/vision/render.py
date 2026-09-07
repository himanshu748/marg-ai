from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import VisionConfig
from .models import Observation, SurveyInstance, SurveyResult
from .privacy import SensitiveRegionDetector
from .privacy import redact as redact_frame


def _active_observation(
    instance: SurveyInstance,
    frame_idx: int,
    hold_frames: int = 15,
) -> Observation | None:
    observations = [
        observation
        for observation in instance.observations
        if observation.frame_idx <= frame_idx
    ]
    if not observations:
        return None
    latest = max(observations, key=lambda observation: observation.frame_idx)
    if frame_idx - latest.frame_idx > hold_frames:
        return None
    return latest


def _severity_color(severity: int) -> tuple[int, int, int]:
    if severity >= 4:
        return 0, 0, 255
    if severity == 3:
        return 0, 255, 255
    return 0, 200, 0


def _draw_label(frame: object, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x, max(16, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def render_annotated_video(
    video: Path,
    result: SurveyResult,
    out_path: Path,
    config: VisionConfig,
    redact: bool = True,
) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or result.frames)
    writer: cv2.VideoWriter | None = None
    privacy_detector = SensitiveRegionDetector() if redact else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            frame = source_frame
            if config.target_width > 0 and frame.shape[1] > config.target_width:
                scale = config.target_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (config.target_width, max(1, int(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            active: list[tuple[SurveyInstance, Observation]] = []
            for instance in result.instances:
                observation = _active_observation(instance, frames_written)
                if observation is not None:
                    active.append((instance, observation))
            protected = [tuple(observation.bbox) for _, observation in active]
            if privacy_detector is not None:
                frame, _ = redact_frame(frame, privacy_detector.detect(frame), protect=protected)
            detections = sum(int(value) for value in result.detections.values())
            timestamp = frames_written / max(0.001, fps)
            hud = (
                f"t={timestamp:.1f}s  frame {frames_written + 1}/{total_frames}  "
                f"instances {len(active)}  detections {detections}"
            )
            cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 570), 27), (0, 0, 0), -1)
            _draw_label(frame, hud, (8, 19), (255, 255, 255))
            for instance, observation in active:
                x, y, width, height = [round(value) for value in observation.bbox]
                color = _severity_color(instance.severity)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                _draw_label(
                    frame,
                    f"#{instance.id} {instance.class_name} sev{instance.severity} {instance.fused_conf:.2f}",
                    (x, y - 5),
                    color,
                )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {out_path}")
            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return frames_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an annotated MargAI survey video.")
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-redact", action="store_true")
    args = parser.parse_args()
    result = SurveyResult.model_validate_json(
        (args.survey / "result.json").read_text(encoding="utf-8")
    )
    frames = render_annotated_video(
        args.video,
        result,
        args.out,
        VisionConfig(),
        redact=not args.no_redact,
    )
    print(f"wrote {frames} frames to {args.out}")


if __name__ == "__main__":
    main()
