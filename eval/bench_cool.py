from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.pipeline import run as run_pipeline

from .eval_detector import load_records

DEFAULT_IMAGE_ROOT = Path("/home/ubuntu/assets/rdd2022_india")
DEFAULT_PRICE_A = 0.04937
DEFAULT_PRICE_B = 0.03950


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _build_environment() -> dict[str, Any]:
    build_info = cv2.getBuildInformation()
    baseline = ""
    dispatched = ""
    parallel = ""
    in_cpu_features = False
    for raw_line in build_info.splitlines():
        line = raw_line.strip()
        if line.startswith("Parallel framework:"):
            parallel = line.split(":", 1)[1].strip()
        elif line.startswith("CPU/HW features:"):
            in_cpu_features = True
        elif in_cpu_features and line.startswith("Baseline:"):
            baseline = line.split(":", 1)[1].strip()
        elif in_cpu_features and line.startswith("Dispatched code generation:"):
            dispatched = line.split(":", 1)[1].strip()
        elif in_cpu_features and line and not line.startswith(("Baseline:", "Dispatched")):
            in_cpu_features = False
    packages = importlib.metadata.packages_distributions()
    cv2_packages = packages.get("cv2", [])
    opencv_package = next(
        (
            name
            for name in cv2_packages
            if name.lower() in {
                "opencv-python",
                "opencv-python-headless",
                "opencv-contrib-python",
                "opencv-contrib-python-headless",
            }
        ),
        cv2_packages[0] if cv2_packages else "unknown",
    )
    return {
        "platform_machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "opencv_package": opencv_package,
        "parallel_framework": parallel,
        "cpu_hw_features_baseline": baseline,
        "cpu_hw_features_dispatched": dispatched,
        "cool_present": "COOL" in build_info,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _run_video(video: Path, repeats: int, config: VisionConfig) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="marg-bench-warmup-") as temporary:
        run_pipeline(video, out_dir=Path(temporary) / "warmup", config=config)
    wall_times: list[float] = []
    results: list[Any] = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory(prefix=f"marg-bench-{repeat}-") as temporary:
            started = time.perf_counter()
            result = run_pipeline(video, out_dir=Path(temporary) / "run", config=config)
            wall_times.append(time.perf_counter() - started)
            results.append(result)
    wall_s = _median(wall_times)
    result = results[0]
    frames = int(result.frames)
    processed_frames = int(result.processed_frames)
    stage_keys = [
        key
        for key in result.metrics
        if key.startswith("stage_") and key.endswith("_s")
    ]
    stage_medians = {
        key: _median([float(item.metrics.get(key, 0.0)) for item in results])
        for key in stage_keys
    }
    return {
        "video": str(video),
        "name": video.name,
        "repeats": repeats,
        "frames": frames,
        "processed_frames": processed_frames,
        "keyframes": int(result.keyframes),
        "wall_s": wall_s,
        "fps_source": frames / max(0.001, wall_s),
        "fps_processed": processed_frames / max(0.001, wall_s),
        "stage_metrics": stage_medians,
        **stage_medians,
    }


def _image_root(root: Path) -> tuple[Path, Path]:
    candidates = [
        (root / "eval_slice" / "images", root / "eval_slice" / "labels"),
        (root / "subset" / "images", root / "subset" / "labels"),
        (root / "images", root / "labels"),
    ]
    for images, labels in candidates:
        if images.is_dir() and labels.is_dir():
            return images, labels
    raise FileNotFoundError(f"Could not find image/label directories under {root}")


def _benchmark_detector(root: Path) -> dict[str, Any]:
    images_dir, labels_dir = _image_root(root)
    records = load_records(images_dir, labels_dir)[:200]
    config = VisionConfig(conf_thresh=0.001)
    detector = DNNDetector(config.model_file(Path(__file__).resolve().parents[1]), config)
    if not records:
        return {"images": 0, "median_ms": 0.0, "p95_ms": 0.0, "images_per_second": 0.0}
    first = cv2.imread(str(records[0].path))
    if first is None:
        return {"images": 0, "median_ms": 0.0, "p95_ms": 0.0, "images_per_second": 0.0}
    detector.detect(first)
    latencies: list[float] = []
    for record in records:
        image = cv2.imread(str(record.path))
        if image is None:
            continue
        started = time.perf_counter()
        detector.detect(image)
        latencies.append((time.perf_counter() - started) * 1000.0)
    if not latencies:
        return {"images": 0, "median_ms": 0.0, "p95_ms": 0.0, "images_per_second": 0.0}
    return {
        "images": len(latencies),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "images_per_second": 1000.0 / float(np.mean(latencies)),
    }


def run_benchmark(
    label: str,
    videos: list[Path],
    out_path: Path,
    repeats: int = 2,
    image_root: Path = DEFAULT_IMAGE_ROOT,
) -> dict[str, Any]:
    config = VisionConfig()
    result: dict[str, Any] = {
        "label": label,
        "environment": _build_environment(),
        "videos": [_run_video(video, repeats, config) for video in videos],
        "detector": _benchmark_detector(image_root),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _video_map(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name") or item["video"]): item
        for item in benchmark["videos"]
    }


def build_comparison(
    a: dict[str, Any],
    b: dict[str, Any],
    price_a: float,
    price_b: float,
) -> str:
    rows = [
        "# COOL benchmark comparison",
        "",
        (
            "Prices are hourly Fargate prices for 1 vCPU + 2 GB in us-east-1: "
            f"{a['label']} **${price_a:.5f}/h** and {b['label']} **${price_b:.5f}/h**. "
            "Source: [AWS Fargate pricing](https://aws.amazon.com/fargate/pricing/)."
        ),
        "",
        "| Video | Label | Architecture | Source FPS | Median wall (s) | Detector median (ms) | Speedup B/A | Cost / 1,000 source frames |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    videos_a = _video_map(a)
    videos_b = _video_map(b)
    for name in videos_a:
        first = videos_a[name]
        second = videos_b.get(name)
        for benchmark, item, price, speedup in (
            (a, first, price_a, None),
            (b, second, price_b, None if second is None else second["fps_source"] / max(0.001, first["fps_source"])),
        ):
            if item is None:
                continue
            architecture = benchmark["environment"]["platform_machine"]
            cost = price * float(item["wall_s"]) / 3600.0 / max(1, int(item["frames"])) * 1000
            speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
            rows.append(
                f"| {name} | {benchmark['label']} | {architecture} | "
                f"{item['fps_source']:.2f} | {item['wall_s']:.2f} | "
                f"{benchmark['detector']['median_ms']:.2f} | {speedup_text} | ${cost:.6f} |"
            )
    rows.extend(
        [
            "",
            "| Detector benchmark | Label | Architecture | Images | Median (ms) | P95 (ms) | Images/s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for benchmark in (a, b):
        detector = benchmark["detector"]
        rows.append(
            f"| RDD2022 first 200 images | {benchmark['label']} | "
            f"{benchmark['environment']['platform_machine']} | {detector['images']} | "
            f"{detector['median_ms']:.2f} | {detector['p95_ms']:.2f} | "
            f"{detector['images_per_second']:.2f} |"
        )
    return "\n".join(rows) + "\n"


def compare_benchmarks(
    first_path: Path,
    second_path: Path,
    price_a: float,
    price_b: float,
    out_path: Path,
) -> str:
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    markdown = build_comparison(first, second, price_a, price_b)
    out_path.write_text(markdown, encoding="utf-8")
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MargAI against OpenCV COOL.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--video", type=Path, action="append", required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--repeats", type=int, default=2)
    run_parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_ROOT)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("first", type=Path)
    compare_parser.add_argument("second", type=Path)
    compare_parser.add_argument("--price-a", type=float, default=DEFAULT_PRICE_A)
    compare_parser.add_argument("--price-b", type=float, default=DEFAULT_PRICE_B)
    compare_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        print(json.dumps(run_benchmark(args.label, args.video, args.out, args.repeats, args.images), indent=2))
    else:
        print(compare_benchmarks(args.first, args.second, args.price_a, args.price_b, args.out))


if __name__ == "__main__":
    main()
