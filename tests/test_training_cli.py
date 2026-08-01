import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts" / "rsl_rl"


def _import_script_without_runtime(script_name: str):
    """Import a CLI module while rejecting simulator and WandB imports."""
    real_import = builtins.__import__

    def reject_runtime_imports(name, *args, **kwargs):
        if name == "wandb" or name.startswith(("isaaclab", "isaacsim", "omni", "carb")):
            raise AssertionError(f"parser import loaded runtime dependency: {name}")
        return real_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", SCRIPTS_DIR / f"{script_name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        patch.object(sys, "path", [str(SCRIPTS_DIR), *sys.path]),
        patch("builtins.__import__", side_effect=reject_runtime_imports),
    ):
        spec.loader.exec_module(module)
    return module


def test_training_parser_import_does_not_load_simulator_or_wandb():
    module = _import_script_without_runtime("train")

    assert callable(module.create_parser)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--motion_file", "motion.npz", "--registry_name", "team/project/walk"],
    ],
)
def test_training_parser_requires_exactly_one_motion_source(arguments):
    parser = _import_script_without_runtime("train").create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


@pytest.mark.parametrize(
    ("arguments", "motion_file", "registry_name"),
    [
        (["--motion_file", "motion.npz"], "motion.npz", None),
        (["--registry_name", "team/project/walk"], None, "team/project/walk"),
    ],
)
def test_training_parser_accepts_either_motion_source(arguments, motion_file, registry_name):
    parser = _import_script_without_runtime("train").create_parser()

    args = parser.parse_args(arguments)

    assert args.motion_file == motion_file
    assert args.registry_name == registry_name
    assert args.logger == "tensorboard"


def test_training_parser_accepts_explicit_log_directory():
    module = _import_script_without_runtime("train")
    parser = module.create_parser()

    args = parser.parse_args(["--motion_file", "motion.npz", "--log_dir", "artifacts/runs/run-1/tensorboard"])

    assert args.log_dir == "artifacts/runs/run-1/tensorboard"
    assert module._resolve_log_dir(args, SimpleNamespace(experiment_name="tracking", run_name="ignored")) == (
        "artifacts/runs/run-1/tensorboard"
    )


def test_training_application_closes_after_success():
    module = _import_script_without_runtime("train")
    events = []
    app = SimpleNamespace(close=lambda: events.append("close"))

    with patch.object(module, "_run_training", side_effect=lambda args, application: events.append("train")):
        module._run_application(SimpleNamespace(), app)

    assert events == ["train", "close"]


def test_training_application_does_not_hide_training_error_during_cleanup():
    module = _import_script_without_runtime("train")
    app = SimpleNamespace(close=lambda: pytest.fail("close must not hide the training error"))

    with (
        patch.object(module, "_run_training", side_effect=ImportError("Isaac API mismatch")),
        pytest.raises(ImportError, match="Isaac API mismatch"),
    ):
        module._run_application(SimpleNamespace(), app)


def test_training_uses_isaac51_yaml_configuration_snapshots_only():
    source = (SCRIPTS_DIR / "train.py").read_text()

    assert "dump_yaml" in source
    assert "dump_pickle" not in source
    assert "handle_deprecated_rsl_rl_cfg" in source


def test_play_parser_import_does_not_load_simulator_or_wandb():
    module = _import_script_without_runtime("play")

    assert callable(module.create_parser)


def test_play_parser_accepts_complete_local_pair():
    parser = _import_script_without_runtime("play").create_parser()

    args = parser.parse_args(["--checkpoint", "model.pt", "--motion_file", "motion.npz"])

    assert args.checkpoint == "model.pt"
    assert args.motion_file == "motion.npz"
    assert args.wandb_path is None


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--checkpoint", "model.pt"],
        ["--motion_file", "motion.npz"],
        ["--wandb_path", "team/project/run", "--checkpoint", "model.pt", "--motion_file", "motion.npz"],
    ],
)
def test_play_parser_rejects_missing_or_mixed_local_pair(arguments):
    parser = _import_script_without_runtime("play").create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


def test_play_parser_accepts_explicit_wandb_compatibility_mode():
    parser = _import_script_without_runtime("play").create_parser()

    args = parser.parse_args(["--wandb_path", "team/project/run"])

    assert args.wandb_path == "team/project/run"
    assert args.checkpoint is None
    assert args.motion_file is None
