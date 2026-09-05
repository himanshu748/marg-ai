from pathlib import Path

from marg.vision.config import VisionConfig
from marg.vision.geo import GeoTrack


def test_geo_synthetic_and_gpx(tmp_path: Path) -> None:
    config = VisionConfig()
    synthetic = GeoTrack.from_path(None, config)
    assert synthetic.source == "synthetic"
    assert synthetic.sample(0.0) == (config.synthetic_start_lat, config.synthetic_start_lon)
    gpx = tmp_path / "track.gpx"
    gpx.write_text(
        """<?xml version="1.0"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
<trk><trkseg>
<trkpt lat="28.6" lon="77.2"><time>2026-01-01T00:00:00Z</time></trkpt>
<trkpt lat="28.601" lon="77.201"><time>2026-01-01T00:00:01Z</time></trkpt>
</trkseg></trk>
</gpx>""",
        encoding="utf-8",
    )
    parsed = GeoTrack.from_path(gpx, config)
    assert parsed.source == "gpx"
    assert parsed.sample(0.0) == (28.6, 77.2)
