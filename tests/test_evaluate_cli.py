import builtins
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
        [
            "--checkpoint",
            "model.pt",
            "--motion_file",
            "motion.npz",
            "--seed",
            "42",
            "--output_file",
            "result.json",
        ],
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
            "--motion_id",
            "jump",
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


def test_segment_start_frames_are_seeded_and_leave_a_full_horizon():
    module = _import_evaluate_without_runtime()

    first = module._segment_start_frames(100, motion_frames=13065, horizon_steps=500, seed=42)
    repeated = module._segment_start_frames(100, motion_frames=13065, horizon_steps=500, seed=42)
    different = module._segment_start_frames(100, motion_frames=13065, horizon_steps=500, seed=43)

    assert first.tolist() == repeated.tolist()
    assert first.tolist() != different.tolist()
    assert len(set(first.tolist())) == 100
    assert int(first.min()) >= 0
    assert int(first.max()) <= 13065 - 500


def test_disable_manager_terms_preserves_dataclass_and_configclass_like_containers():
    module = _import_evaluate_without_runtime()

    @dataclass
    class DataclassManagerCfg:
        reset: object
        interval: object

    class ConfigclassLikeManagerCfg:
        def __init__(self):
            self.curriculum_term = object()

    containers = (
        DataclassManagerCfg(reset=object(), interval=object()),
        ConfigclassLikeManagerCfg(),
    )

    for container in containers:
        identity = id(container)
        result = module._disable_manager_terms(container)

        assert result is container
        assert id(result) == identity
        assert result is not None
        assert vars(result)
        assert all(value is None for value in vars(result).values())


class _FakeBaseEnv:
    def __init__(self):
        self.device = "cpu"
        self.reset_ids = []
        self.history_reset_ids = []
        self.events = []
        self.scene = SimpleNamespace(write_data_to_sim=lambda: self.events.append("write"))
        self.sim = SimpleNamespace(forward=lambda: self.events.append("forward"))
        self.observation_manager = SimpleNamespace(
            reset=lambda env_ids: (
                self.history_reset_ids.append(env_ids.tolist()),
                self.events.append("history_reset"),
            ),
            compute=lambda **_: (
                self.events.append("compute"),
                {"policy": np.array([[1.0]])},
            )[1],
        )

    def _reset_idx(self, env_ids):
        self.reset_ids.append(env_ids.tolist())


class _FakeCommand:
    def __init__(self):
        self.start_ids = []

    def reset_to_start(self, env_ids):
        self.start_ids.append(env_ids.tolist())

    def reset_to_frame(self, env_ids, frame_ids):
        self.start_ids.append((env_ids.tolist(), frame_ids.tolist()))

    def update_reference_alignment(self):
        return None


def test_initial_segment_reset_uses_exact_start_frames():
    module = _import_evaluate_without_runtime()
    env = _FakeBaseEnv()
    env.num_envs = 2
    command = _FakeCommand()

    module._reset_all_to_frames(env, command, np.array([10, 20]))

    assert command.start_ids == [([0, 1], [10, 20])]
    assert env.events == ["write", "forward", "history_reset", "compute"]


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
    assert env.events == ["write", "forward", "history_reset", "compute"]
    assert observations["policy"].tolist() == [[1.0]]


def test_initial_reset_starts_every_environment_at_frame_zero():
    module = _import_evaluate_without_runtime()
    command = _FakeCommand()
    history_reset_ids = []
    events = []
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene=SimpleNamespace(write_data_to_sim=lambda: events.append("write")),
        sim=SimpleNamespace(forward=lambda: events.append("forward")),
        observation_manager=SimpleNamespace(
            reset=lambda env_ids: (
                history_reset_ids.append(env_ids.tolist()),
                events.append("history_reset"),
            ),
            compute=lambda **_: (
                events.append("compute"),
                {"policy": np.array([[2.0]])},
            )[1],
        ),
    )

    observations = module._reset_all_to_start(env, command)

    assert command.start_ids == [[0, 1, 2]]
    assert history_reset_ids == [[0, 1, 2]]
    assert events == ["write", "forward", "history_reset", "compute"]
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


def test_evaluator_migrates_current_rsl_rl_config_before_runner_construction(tmp_path):
    module = _import_evaluate_without_runtime()
    checkpoint = tmp_path / "model.pt"
    motion = tmp_path / "motion.npz"
    checkpoint.write_bytes(b"checkpoint")
    motion.write_bytes(b"motion")
    events = []

    class RunnerReached(Exception):
        pass

    class AgentCfg:
        seed = 0
        device = "cpu"

        def __init__(self):
            self.actor = {}

        def to_dict(self):
            events.append("to_dict")
            return {"actor": dict(self.actor)}

    agent_cfg = AgentCfg()
    event_config = SimpleNamespace(reset=object(), interval=object())
    curriculum_config = SimpleNamespace(progress=object())
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=0),
        seed=0,
        events=event_config,
        curriculum=curriculum_config,
        commands=SimpleNamespace(motion=SimpleNamespace(motion_file="old.npz")),
    )
    env = SimpleNamespace(close=lambda: events.append("close"))

    def make_env(*_args, **_kwargs):
        assert env_cfg.events is event_config
        assert env_cfg.curriculum is curriculum_config
        assert all(value is None for value in vars(event_config).values())
        assert all(value is None for value in vars(curriculum_config).values())
        return env

    def hydra_task_config(_task, _entry_point):
        def decorate(function):
            return lambda: function(env_cfg, agent_cfg)

        return decorate

    def migrate(config, version):
        assert version == "2.3.3"
        events.append("migrate")
        config.actor["class_name"] = "ActorCritic"
        return config

    def runner(_env, config, **_kwargs):
        events.append("runner")
        assert config["actor"]["class_name"] == "ActorCritic"
        raise RunnerReached

    fake_modules = {
        "gymnasium": SimpleNamespace(make=make_env),
        "whole_body_tracking.tasks": ModuleType("whole_body_tracking.tasks"),
        "isaaclab_rl": ModuleType("isaaclab_rl"),
        "isaaclab_rl.rsl_rl": SimpleNamespace(
            RslRlVecEnvWrapper=lambda value: SimpleNamespace(unwrapped=value),
            handle_deprecated_rsl_rl_cfg=migrate,
        ),
        "isaaclab_tasks": ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.utils": ModuleType("isaaclab_tasks.utils"),
        "isaaclab_tasks.utils.hydra": SimpleNamespace(hydra_task_config=hydra_task_config),
        "rsl_rl": ModuleType("rsl_rl"),
        "rsl_rl.runners": SimpleNamespace(OnPolicyRunner=runner),
        "tensordict": SimpleNamespace(TensorDict=dict),
        "whole_body_tracking.utils.evaluation": SimpleNamespace(
            EpisodeAccumulator=object,
            atomic_write_evaluation_json=lambda *_args, **_kwargs: None,
        ),
    }
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        motion_file=str(motion),
        output_file=str(tmp_path / "result.json"),
        episodes=1,
        num_envs=1,
        seed=42,
        task="Tracking-Flat-G1-v0",
        motion_id="walk_run",
    )

    with (
        patch.dict(sys.modules, fake_modules),
        patch("importlib.metadata.version", return_value="2.3.3"),
        pytest.raises(RunnerReached),
    ):
        module._run_evaluation(args, SimpleNamespace())

    assert events == ["migrate", "to_dict", "runner", "close"]


def test_evaluator_step_keeps_terminal_reset_state_mutable(tmp_path):
    module = _import_evaluate_without_runtime()
    checkpoint = tmp_path / "model.pt"
    motion = tmp_path / "motion.npz"
    checkpoint.write_bytes(b"checkpoint")
    motion.write_bytes(b"motion")
    events = []

    import torch

    class Command:
        def __init__(self):
            self.time_steps = torch.tensor([0])
            self.motion = SimpleNamespace(time_step_total=2)
            self.metrics = {"error_body_pos": torch.tensor([0.25])}

        def reset_to_frame(self, _env_ids, _frame_ids):
            events.append("reset_to_frame")

        def update_reference_alignment(self):
            events.append("align")

        def _update_metrics(self):
            return None

    class TerminationManager:
        terminated = torch.tensor([False])
        time_outs = torch.tensor([True])

        def compute(self):
            return self.terminated

    command = Command()

    class Env:
        device = "cpu"
        num_envs = 1
        max_episode_length = 1

        def __init__(self):
            self.command_manager = SimpleNamespace(get_term=lambda _name: command)
            self.termination_manager = TerminationManager()
            self.scene = SimpleNamespace(write_data_to_sim=lambda: events.append("write"))
            self.sim = SimpleNamespace(forward=lambda: events.append("forward"))
            self.observation_manager = SimpleNamespace(
                reset=lambda _env_ids: None,
                compute=lambda **_kwargs: {"policy": torch.zeros((1, 1))},
            )
            self.mutable_state = None

        def _reset_idx(self, _env_ids):
            self.mutable_state.add_(1)
            events.append(("reset_value", self.mutable_state.item()))

        def close(self):
            events.append("close")

    env = Env()

    class Wrapper:
        def __init__(self, value):
            self.unwrapped = value

        def step(self, _actions):
            events.append(
                ("step_context", torch.is_inference_mode_enabled(), torch.is_grad_enabled())
            )
            env.mutable_state = torch.zeros(1)
            env.mutable_state.add_(1)
            events.append(("reset_value", env.mutable_state.item()))
            env.termination_manager.compute()
            return {"policy": torch.zeros((1, 1))}, None, None, None

    def policy(_observations):
        events.append(("policy_context", torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        return torch.zeros((1, 1))

    class Runner:
        def __init__(self, *_args, **_kwargs):
            return None

        def load(self, _checkpoint):
            return None

        def get_inference_policy(self, **_kwargs):
            return policy

    class Accumulator:
        def __init__(self, *_args):
            self.done = False

        def update(self, *_args):
            self.done = True

        def reset(self, _env_ids):
            return None

        def result(self):
            return SimpleNamespace(episodes=1, completed=1, falls=0, tracking_errors=(0.25,))

    class TensorDict(dict):
        def __init__(self, values, batch_size):
            assert batch_size == [1]
            super().__init__(values)

    agent_cfg = SimpleNamespace(seed=0, device="cpu", to_dict=dict)
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=0),
        seed=0,
        events=SimpleNamespace(randomize=object()),
        curriculum=SimpleNamespace(),
        commands=SimpleNamespace(motion=SimpleNamespace(motion_file="old.npz")),
    )

    def hydra_task_config(_task, _entry_point):
        def decorate(function):
            return lambda: function(env_cfg, agent_cfg)

        return decorate

    fake_modules = {
        "gymnasium": SimpleNamespace(make=lambda *_args, **_kwargs: env),
        "whole_body_tracking.tasks": ModuleType("whole_body_tracking.tasks"),
        "isaaclab_rl": ModuleType("isaaclab_rl"),
        "isaaclab_rl.rsl_rl": SimpleNamespace(
            RslRlVecEnvWrapper=Wrapper,
            handle_deprecated_rsl_rl_cfg=lambda config, _version: config,
        ),
        "isaaclab_tasks": ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.utils": ModuleType("isaaclab_tasks.utils"),
        "isaaclab_tasks.utils.hydra": SimpleNamespace(hydra_task_config=hydra_task_config),
        "rsl_rl": ModuleType("rsl_rl"),
        "rsl_rl.runners": SimpleNamespace(OnPolicyRunner=Runner),
        "tensordict": SimpleNamespace(TensorDict=TensorDict),
        "whole_body_tracking.utils.evaluation": SimpleNamespace(
            EpisodeAccumulator=Accumulator,
            atomic_write_evaluation_json=lambda *_args, **_kwargs: None,
        ),
    }
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        motion_file=str(motion),
        output_file=str(tmp_path / "result.json"),
        episodes=1,
        num_envs=1,
        seed=42,
        task="Tracking-Flat-G1-v0",
        motion_id="walk_run",
    )

    with (
        patch.dict(sys.modules, fake_modules),
        patch("importlib.metadata.version", return_value="2.3.3"),
    ):
        module._run_evaluation(args, SimpleNamespace(is_running=lambda: True))

    assert ("policy_context", False, False) in events
    assert ("step_context", False, False) in events
    assert ("reset_value", 1.0) in events
    assert events[-1] == "close"
