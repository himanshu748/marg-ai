from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    fused_conf: float = 0.0


@dataclass(slots=True)
class FramePacket:
    index: int
    timestamp_s: float
    frame: object


@dataclass(slots=True)
class QualityResult:
    accepted: bool
    blur_variance: float
    clipped_fraction: float


@dataclass(slots=True)
class Keyframe:
    keyframe_id: int
    frame_index: int
    timestamp_s: float
    frame: object


class Observation(BaseModel):
    frame_idx: int
    t_s: float
    bbox: list[float]
    class_name: str
    conf: float
    fused_conf: float


@dataclass(slots=True)
class TrackUpdate:
    instance_id: int
    detection: Detection
    observation: Observation
    best_observation: Observation | None = None


class SurveyInstance(BaseModel):
    id: int
    bbox: list[float]
    class_name: str
    confidence: float
    fused_conf: float
    severity: int
    area_m2: float
    lat: float
    lon: float
    keyframe_ids: list[int] = Field(default_factory=list)
    frame_confs: list[float] = Field(default_factory=list)
    frame_classes: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    best_frame_idx: int = -1
    evidence_path: str | None = None


class Segment(BaseModel):
    id: int
    instance_ids: list[int]
    lat: float
    lon: float
    radius_m: float


class SurveySummary(BaseModel):
    id: str
    video: str
    geo_source: str


class SurveyResult(BaseModel):
    survey_id: str
    video: str
    gpx: str | None = None
    geo_source: str
    frames: int
    processed_frames: int
    keyframes: int
    detections: dict[str, int]
    surveys: list[SurveySummary] = Field(default_factory=list)
    instances: list[SurveyInstance] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    keyframe_paths: list[str] = Field(default_factory=list)
    crop_paths: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
