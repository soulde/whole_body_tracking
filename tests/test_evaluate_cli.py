import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts" / "rsl_rl"


def _import_evaluate_without_runtime():
    """Import the evaluator while rejecting simulator and WandB imports."""
    real_import = builtins.__import__

    def reject_runtime_imports(name, *args, **kwargs):
        if name == "wandb" or name.startswith(("isaaclab", "isaacsim", "omni", "carb")):
            raise AssertionError(f"parser import loaded runtime dependency: {name}")
        return real_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("evaluate_under_test", SCRIPTS_DIR / "evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        patch.object(sys, "path", [str(SCRIPTS_DIR), *sys.path]),
        patch("builtins.__import__", side_effect=reject_runtime_imports),
    ):
        spec.loader.exec_module(module)
    return module


def test_evaluate_parser_import_does_not_load_simulator_or_wandb():
    module = _import_evaluate_without_runtime()

    assert callable(module.create_parser)


def test_evaluate_parser_requires_local_artifacts_seed_and_output():
    parser = _import_evaluate_without_runtime().create_parser()

    for arguments in (
        [],
        ["--motion_file", "motion.npz", "--seed", "42", "--output_file", "result.json"],
        ["--checkpoint", "model.pt", "--seed", "42", "--output_file", "result.json"],
        ["--checkpoint", "model.pt", "--motion_file", "motion.npz", "--output_file", "result.json"],
        ["--checkpoint", "model.pt", "--motion_file", "motion.npz", "--seed", "42"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(arguments)


def test_evaluate_parser_uses_clean_evaluation_defaults():
    parser = _import_evaluate_without_runtime().create_parser()

    args = parser.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--motion_file",
            "motion.npz",
            "--seed",
            "42",
            "--output_file",
            "result.json",
        ]
    )

    assert args.episodes == 100
    assert args.num_envs == 100
    assert args.task == "Tracking-Flat-G1-v0"


class _FakeBaseEnv:
    def __init__(self):
        self.device = "cpu"
        self.reset_ids = []
        self.history_reset_ids = []
        self.observation_manager = SimpleNamespace(
            reset=lambda env_ids: self.history_reset_ids.append(env_ids.tolist()),
            compute=lambda **_: {"policy": np.array([[1.0]])},
        )

    def _reset_idx(self, env_ids):
        self.reset_ids.append(env_ids.tolist())


class _FakeCommand:
    def __init__(self):
        self.start_ids = []

    def reset_to_start(self, env_ids):
        self.start_ids.append(env_ids.tolist())


def test_terminal_reset_restarts_failed_and_completed_environments_at_frame_zero():
    module = _import_evaluate_without_runtime()
    env = _FakeBaseEnv()
    command = _FakeCommand()

    observations = module._reset_terminal_environments(
        env,
        command,
        failed_ids=np.array([3, 1]),
        completed_ids=np.array([4, 2]),
    )

    assert env.reset_ids == [[2, 4]]
    assert command.start_ids == [[1, 2, 3, 4]]
    assert env.history_reset_ids == [[1, 2, 3, 4]]
    assert observations["policy"].tolist() == [[1.0]]


def test_initial_reset_starts_every_environment_at_frame_zero():
    module = _import_evaluate_without_runtime()
    command = _FakeCommand()
    history_reset_ids = []
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        observation_manager=SimpleNamespace(
            reset=lambda env_ids: history_reset_ids.append(env_ids.tolist()),
            compute=lambda **_: {"policy": np.array([[2.0]])},
        ),
    )

    observations = module._reset_all_to_start(env, command)

    assert command.start_ids == [[0, 1, 2]]
    assert history_reset_ids == [[0, 1, 2]]
    assert observations["policy"].tolist() == [[2.0]]


def test_metric_capture_preserves_post_physics_error_before_automatic_reset():
    module = _import_evaluate_without_runtime()

    class _Command:
        def __init__(self):
            self.physical_error = np.array([0.1, 0.2])
            self.metrics = {"error_body_pos": self.physical_error.copy()}

        def _update_metrics(self):
            self.metrics["error_body_pos"] = self.physical_error.copy()

    command = _Command()

    class _TerminationManager:
        def compute(self):
            command.physical_error = np.array([0.7, 0.9])
            return np.array([False, True])

    manager = _TerminationManager()
    captured = module._install_terminal_metric_capture(manager, command)

    assert manager.compute().tolist() == [False, True]
    command.metrics["error_body_pos"][:] = 0.0  # model Isaac's post-reset metric update
    assert captured().tolist() == [0.7, 0.9]


def test_unexpected_timeout_fails_but_same_step_completion_has_priority():
    module = _import_evaluate_without_runtime()

    module._require_no_unexpected_timeouts(
        timeout_mask=np.array([False, True]),
        failed_mask=np.array([False, False]),
        completed_mask=np.array([False, True]),
    )
    module._require_no_unexpected_timeouts(
        timeout_mask=np.array([False, True]),
        failed_mask=np.array([False, True]),
        completed_mask=np.array([False, False]),
    )
    with pytest.raises(RuntimeError, match=r"unexpected timeout.*1"):
        module._require_no_unexpected_timeouts(
            timeout_mask=np.array([False, True]),
            failed_mask=np.array([False, False]),
            completed_mask=np.array([False, False]),
        )


def test_snapshot_file_hashes_the_immutable_bytes_consumed_by_evaluation(tmp_path):
    module = _import_evaluate_without_runtime()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"first")
    snapshot, digest = module._snapshot_file(checkpoint, tmp_path / "snapshots")
    checkpoint.write_bytes(b"second")

    assert snapshot.read_bytes() == b"first"
    assert digest == "a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e"
