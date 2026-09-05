import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import gpxpy

from .config import VisionConfig
from .models import Segment, SurveyInstance


@dataclass(slots=True)
class GeoPoint:
    timestamp_s: float
    lat: float
    lon: float


class GeoTrack:
    def __init__(self, points: list[GeoPoint], source: str, config: VisionConfig) -> None:
        self.points = sorted(points, key=lambda point: point.timestamp_s)
        self.source = source
        self.config = config

    @classmethod
    def from_path(cls, path: str | Path | None, config: VisionConfig) -> "GeoTrack":
        if path is None:
            return cls([], "synthetic", config)
        suffix = Path(path).suffix.lower()
        if suffix == ".srt":
            return cls(_parse_srt(Path(path)), "srt", config)
        with Path(path).open("r", encoding="utf-8") as handle:
            gpx = gpxpy.parse(handle)
        points: list[GeoPoint] = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if point.time is not None:
                        points.append(
                            GeoPoint(_datetime_seconds(point.time), point.latitude, point.longitude)
                        )
        return cls(points, "gpx", config)

    def sample(self, video_timestamp_s: float) -> tuple[float, float]:
        if not self.points:
            distance_km = self.config.synthetic_speed_kmh * max(0.0, video_timestamp_s) / 3600.0
            lat_delta = distance_km / 111.32
            return (
                self.config.synthetic_start_lat + lat_delta,
                self.config.synthetic_start_lon,
            )
        target = self.points[0].timestamp_s + video_timestamp_s + self.config.timestamp_offset_s
        point = min(self.points, key=lambda candidate: abs(candidate.timestamp_s - target))
        return point.lat, point.lon


def cluster_instances(instances: list[SurveyInstance], radius_m: float) -> list[Segment]:
    segments: list[Segment] = []
    for instance in instances:
        assigned = False
        for segment in segments:
            if haversine_m(instance.lat, instance.lon, segment.lat, segment.lon) <= radius_m:
                segment.instance_ids.append(instance.id)
                assigned = True
                break
        if not assigned:
            segments.append(
                Segment(
                    id=len(segments),
                    instance_ids=[instance.id],
                    lat=instance.lat,
                    lon=instance.lon,
                    radius_m=radius_m,
                )
            )
    return segments


def haversine_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    d_lat = radians(lat_b - lat_a)
    d_lon = radians(lon_b - lon_a)
    value = sin(d_lat / 2) ** 2 + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(d_lon / 2) ** 2
    return 6_371_000.0 * 2.0 * asin(sqrt(value))


def _datetime_seconds(value: datetime) -> float:
    return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()


def _parse_srt(path: Path) -> list[GeoPoint]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    points: list[GeoPoint] = []
    pattern = re.compile(
        r"(?P<start>\d\d:\d\d:\d\d[,.]\d+).*?(?P<lat>-?\d+\.\d+)[,\s]+(?P<lon>-?\d+\.\d+)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        timestamp = _clock_seconds(match.group("start"))
        points.append(GeoPoint(timestamp, float(match.group("lat")), float(match.group("lon"))))
    return points


def _clock_seconds(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)
