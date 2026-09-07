import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from marg.store.local import LocalStore, atomic_write_json


def test_failed_atomic_replace_preserves_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    atomic_write_json(target, {"version": 1})

    def fail_replace(*args):
        raise OSError("disk error")

    monkeypatch.setattr("marg.store.local.os.replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_json(target, {"version": 2})
    assert json.loads(target.read_text()) == {"version": 1}
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_atomic_writes_always_publish_complete_json(tmp_path):
    target = tmp_path / "result.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda value: atomic_write_json(target, {"value": value, "payload": "x" * 2000}), range(20)))
    assert json.loads(target.read_text())["value"] in range(20)
    assert list(tmp_path.iterdir()) == [target]


def test_store_rejects_path_escape_and_nonfinite_json(tmp_path):
    store = LocalStore(tmp_path)
    with pytest.raises(ValueError):
        store.save_json("../escape", "result.json", {})
    with pytest.raises(ValueError):
        store.save_json("safe", "../escape.json", {})
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "nan.json", {"value": float("nan")})
