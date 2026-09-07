from eval.bench_cool import build_comparison


def test_build_comparison_contains_labels_and_speedup() -> None:
    first = {
        "label": "x86",
        "environment": {"platform_machine": "x86_64"},
        "videos": [
            {"name": "clip.webm", "frames": 100, "wall_s": 10.0, "fps_source": 10.0}
        ],
        "detector": {"images": 2, "median_ms": 20.0, "p95_ms": 25.0, "images_per_second": 50.0},
    }
    second = {
        "label": "arm",
        "environment": {"platform_machine": "aarch64"},
        "videos": [
            {"name": "clip.webm", "frames": 100, "wall_s": 5.0, "fps_source": 20.0}
        ],
        "detector": {"images": 2, "median_ms": 10.0, "p95_ms": 12.0, "images_per_second": 100.0},
    }
    markdown = build_comparison(first, second, 0.05, 0.04)
    assert "x86" in markdown
    assert "arm" in markdown
    assert "2.00x" in markdown
