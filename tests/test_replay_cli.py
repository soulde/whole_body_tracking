import builtins
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "replay_npz.py"


def _import_without_simulator_or_wandb():
    real_import = builtins.__import__

    def reject_runtime_imports(name, *args, **kwargs):
        if name == "wandb" or name.startswith(("isaaclab", "isaacsim", "omni", "carb")):
            raise AssertionError(f"parser import loaded runtime dependency: {name}")
        return real_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("replay_npz_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch("builtins.__import__", side_effect=reject_runtime_imports):
        spec.loader.exec_module(module)
    return module


def test_parser_import_does_not_load_simulator_or_wandb():
    module = _import_without_simulator_or_wandb()

    assert callable(module.create_parser)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--motion_file", "motion.npz", "--registry_name", "team/project/walk"],
    ],
)
def test_parser_requires_exactly_one_motion_source(arguments):
    parser = _import_without_simulator_or_wandb().create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


@pytest.mark.parametrize(
    ("arguments", "motion_file", "registry_name"),
    [
        (["--motion_file", "motion.npz"], "motion.npz", None),
        (["--registry_name", "team/project/walk"], None, "team/project/walk"),
    ],
)
def test_parser_accepts_either_motion_source(arguments, motion_file, registry_name):
    parser = _import_without_simulator_or_wandb().create_parser()

    args = parser.parse_args(arguments)

    assert args.motion_file == motion_file
    assert args.registry_name == registry_name
