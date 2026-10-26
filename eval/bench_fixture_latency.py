"""Reproducible CPU detector timing on committed sample keyframes, without APIs."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "demo").glob("*/keyframes/*.jpg"))
    if not paths:
        raise RuntimeError("No committed demo keyframes found")
    images = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError("A keyframe could not be decoded")
    cv2.setNumThreads(args.threads)
    config = VisionConfig()
    model = config.model_file(root)
    detector = DNNDetector(model, config)
    for _ in range(3):
        detector.detect(images[0])
    timings = []
    counts = []
    for _ in range(args.repeats):
        for image in images:
            started = time.perf_counter()
            detections = detector.detect(image)
            timings.append((time.perf_counter() - started) * 1000)
            counts.append(len(detections))
    cpu = next(
        (line.split(":", 1)[1].strip()
         for line in Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")),
        platform.processor(),
    )
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "scope": "CPU detector latency on existing demo keyframes; no accuracy labels",
        "timed_scope": "letterbox + blob + DNN inference + threshold + NMS; excludes disk IO, model load and privacy pass",
        "environment": {"cpu": cpu, "machine": platform.machine(),
                        "python": platform.python_version(), "opencv": cv2.__version__,
                        "opencv_threads": cv2.getNumThreads()},
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "config": config.model_dump(mode="json"),
        "inputs": [{"path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in paths],
        "unique_images": len(paths), "repeats": args.repeats,
        "warmup_calls": 3, "timed_calls": len(timings),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "images_per_second": 1000 / float(np.mean(timings)),
        "latencies_ms": timings, "detections_per_call": counts,
        "limitations": ["Existing demonstration images, not a held-out evaluation",
                        "Shared local host, not the deployed AWS/Modal environment",
                        "No claim of updated AP, precision, recall, COOL or live-review latency"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "unique_images", "timed_calls", "median_ms", "p95_ms", "images_per_second"
    )}, indent=2))


if __name__ == "__main__":
    main()
