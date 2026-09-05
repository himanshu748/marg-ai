from marg.vision.config import VisionConfig
from marg.vision.models import Detection
from marg.vision.tracker import InstanceTracker


def test_overlapping_boxes_are_one_instance() -> None:
    tracker = InstanceTracker(VisionConfig())
    first = Detection((10, 10, 100, 100), 3, "D40", 0.7, 0.7)
    second = Detection((15, 15, 100, 100), 3, "D40", 0.8, 0.8)
    update_a = tracker.update([first], 0)
    update_b = tracker.update([second], 1)
    assert update_a[0].instance_id == update_b[0].instance_id
    assert len(tracker.finalize()) == 1
