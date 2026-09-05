import numpy as np

from eval.eval_detector import compute_ap


def test_compute_ap_all_point_interpolation() -> None:
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 0.5])
    assert compute_ap(recall, precision) == 0.75
