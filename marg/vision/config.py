from pathlib import Path

from pydantic import BaseModel, Field


class VisionConfig(BaseModel):
    frame_stride: int = 3
    target_width: int = 960
    blur_min_laplacian_var: float = 60.0
    exposure_clip_pct_max: float = 0.25
    keyframe_min_new_ratio: float = 0.35
    keyframe_max_elapsed_s: float = 1.5
    orb_nfeatures: int = 1500
    dnn_input_size: int = 640
    conf_thresh: float = 0.25
    nms_iou: float = 0.45
    classes: list[str] = Field(default_factory=lambda: ["D00", "D10", "D20", "D40"])
    tracker_iou: float = 0.3
    tracker_max_unseen_keyframes: int = 6
    optical_flow_width: int = 480
    timestamp_offset_s: float = 0.0
    synthetic_start_lat: float = 28.6139
    synthetic_start_lon: float = 77.2090
    synthetic_speed_kmh: float = 30.0
    geo_cluster_radius_m: float = 25.0
    model_path: Path = Path("models/rdd_yolov8s.onnx")
    road_plane: list[tuple[float, float]] = Field(
        default_factory=lambda: [
            (0.15, 1.0),
            (0.85, 1.0),
            (0.58, 0.62),
            (0.42, 0.62),
        ]
    )

    def model_file(self, base_dir: Path | None = None) -> Path:
        if self.model_path.is_absolute() or base_dir is None:
            return self.model_path
        return base_dir / self.model_path
