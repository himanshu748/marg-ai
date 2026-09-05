from marg.vision.config import VisionConfig
from marg.vision.models import Detection
from marg.vision.tracker import InstanceTracker


def test_overlapping_boxes_are_one_instance() -> None:
    tracker = InstanceTracker(VisionConfig())
    first = Detection((10, 10, 100, 100), 3, "D40", 0.7, 0.7)
    second = Detection((15, 15, 100, 100), 3, "D40", 0.8, 0.8)
    update_a = tracker.update([first], 0, frame_idx=3, t_s=0.1)
    update_b = tracker.update([second], None, frame_idx=4, t_s=0.2)
    assert update_a[0].instance_id == update_b[0].instance_id
    track = tracker.finalize()[0]
    assert len(track.observations) == 2
    assert track.observations[1].frame_idx == 4
    assert track.best_observation is not None
    assert track.best_observation.bbox == [15.0, 15.0, 100.0, 100.0]
