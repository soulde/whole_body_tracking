import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "local_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("local_artifacts_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LOCAL_ARTIFACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCAL_ARTIFACTS)
atomic_save_npz = LOCAL_ARTIFACTS.atomic_save_npz
atomic_write_json = LOCAL_ARTIFACTS.atomic_write_json
sha256_file = LOCAL_ARTIFACTS.sha256_file


def test_atomic_save_npz_round_trips_arrays_and_returns_hash(tmp_path):
    path = tmp_path / "motion.npz"
    arrays = {
        "positions": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "names": np.array(["left", "right"]),
    }

    digest = atomic_save_npz(path, arrays)

    with np.load(path) as saved:
        assert set(saved.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(saved[name], expected)
    assert digest == sha256_file(path)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["motion.npz"]


def test_atomic_save_npz_does_not_silently_overwrite(tmp_path):
    path = tmp_path / "motion.npz"
    atomic_save_npz(path, {"value": np.array([1])})

    with pytest.raises(FileExistsError):
        atomic_save_npz(path, {"value": np.array([2])})

    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["value"], np.array([1]))


def test_atomic_save_npz_overwrites_when_explicitly_requested(tmp_path):
    path = tmp_path / "motion.npz"
    first_digest = atomic_save_npz(path, {"value": np.array([1])})

    second_digest = atomic_save_npz(path, {"value": np.array([2])}, overwrite=True)

    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["value"], np.array([2]))
    assert second_digest == sha256_file(path)
    assert second_digest != first_digest


def test_atomic_save_npz_cleans_up_temporary_after_save_failure(tmp_path):
    path = tmp_path / "motion.npz"

    def fail_after_partial_write(file_object, **arrays):
        file_object.write(b"partial")
        raise RuntimeError("save failed")

    with patch.object(LOCAL_ARTIFACTS.np, "savez", side_effect=fail_after_partial_write):
        with pytest.raises(RuntimeError, match="save failed"):
            atomic_save_npz(path, {"value": np.array([1])})

    assert list(tmp_path.iterdir()) == []


def test_sha256_file_returns_known_deterministic_digest(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(path) == sha256_file(path)


def test_atomic_write_json_is_deterministic_and_atomic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"z": 1, "a": ["local", True]}

    first_digest = atomic_write_json(first, payload)
    second_digest = atomic_write_json(second, {"a": ["local", True], "z": 1})

    assert json.loads(first.read_text(encoding="utf-8")) == payload
    assert first_digest == sha256_file(first)
    assert first_digest == second_digest
    assert sorted(item.name for item in tmp_path.iterdir()) == ["first.json", "second.json"]


def test_atomic_write_json_requires_explicit_overwrite(tmp_path):
    path = tmp_path / "metadata.json"
    atomic_write_json(path, {"version": 1})

    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"version": 2})

    digest = atomic_write_json(path, {"version": 2}, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert digest == sha256_file(path)
