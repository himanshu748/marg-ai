from marg.vision.severity import score


def test_severity_monotonic() -> None:
    assert score("D40", 0.1) <= score("D40", 0.3) <= score("D40", 0.7)
    assert score("D20", 0.5) <= score("D20", 3.0)
    assert score("D00", 0.05) < score("D00", 0.2)
