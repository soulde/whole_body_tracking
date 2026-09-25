import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("evaluation_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


def test_accumulator_averages_steps_and_classifies_terminal_outcomes():
    accumulator = EVALUATION.EpisodeAccumulator(num_envs=2, target_episodes=2)
    accumulator.update(
        body_error=np.array([0.1, 0.2]),
        failed=np.array([False, False]),
        motion_completed=np.array([False, False]),
    )
    accumulator.update(
        body_error=np.array([0.3, 0.4]),
        failed=np.array([False, True]),
        motion_completed=np.array([True, False]),
    )

    summary = accumulator.result()

    assert summary.episodes == 2
    assert summary.completed == 1
    assert summary.falls == 1
    assert summary.tracking_errors == pytest.approx((0.2, 0.3))


def test_accumulator_requires_reset_before_terminal_environment_contributes_again():
    accumulator = EVALUATION.EpisodeAccumulator(num_envs=1, target_episodes=2)
    accumulator.update(np.array([0.2]), np.array([True]), np.array([False]))
    accumulator.update(np.array([9.0]), np.array([False]), np.array([False]))
    accumulator.reset([0])
    accumulator.update(np.array([0.4]), np.array([False]), np.array([True]))

    assert accumulator.result().tracking_errors == pytest.approx((0.2, 0.4))


def test_accumulator_rejects_invalid_inputs_and_incomplete_result():
    accumulator = EVALUATION.EpisodeAccumulator(num_envs=2, target_episodes=2)

    with pytest.raises(ValueError, match="shape"):
        accumulator.update(np.array([0.1]), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="finite non-negative"):
        accumulator.update(np.array([0.1, np.nan]), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="mutually exclusive"):
        accumulator.update(np.ones(2), np.ones(2, dtype=bool), np.ones(2, dtype=bool))
    with pytest.raises(RuntimeError, match="not complete"):
        accumulator.result()


def test_accumulator_consumes_final_terminal_indices_deterministically():
    accumulator = EVALUATION.EpisodeAccumulator(num_envs=3, target_episodes=2)
    accumulator.update(
        np.array([0.1, 0.2, 0.3]),
        np.array([False, False, False]),
        np.array([True, True, True]),
    )

    assert accumulator.result().tracking_errors == pytest.approx((0.1, 0.2))


def test_atomic_evaluation_json_is_deterministic_and_no_clobber(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"seed": 42, "tracking_errors": [0.2, 0.1]}

    first_digest = EVALUATION.atomic_write_evaluation_json(first, payload)
    second_digest = EVALUATION.atomic_write_evaluation_json(
        second, {"tracking_errors": [0.2, 0.1], "seed": 42}
    )

    assert first_digest == second_digest
    assert json.loads(first.read_text(encoding="utf-8")) == payload
    assert not tuple(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError):
        EVALUATION.atomic_write_evaluation_json(first, payload)


def test_atomic_evaluation_json_concurrent_writers_never_overwrite(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"
    barrier = Barrier(2)
    original_exists = Path.exists

    def synchronized_exists(path):
        if path == destination:
            barrier.wait()
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", synchronized_exists)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(EVALUATION.atomic_write_evaluation_json, destination, {"seed": seed})
            for seed in (42, 43)
        ]
    outcomes = [future.exception() for future in futures]

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
