import builtins
import importlib.util
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "csv_to_npz.py"


def _import_without_simulator_or_wandb():
    real_import = builtins.__import__

    def reject_runtime_imports(name, *args, **kwargs):
        if name == "wandb" or name.startswith(("isaaclab", "isaacsim", "omni", "carb")):
            raise AssertionError(f"parser import loaded runtime dependency: {name}")
        return real_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("csv_to_npz_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch("builtins.__import__", side_effect=reject_runtime_imports):
        spec.loader.exec_module(module)
    return module


def test_parser_import_does_not_load_simulator_or_wandb():
    module = _import_without_simulator_or_wandb()

    assert callable(module.create_parser)


def test_parser_defaults_to_local_motion_artifact():
    parser = _import_without_simulator_or_wandb().create_parser()

    args = parser.parse_args(["--input_file", "walk.csv", "--output_name", "walk"])

    assert args.output_file == "artifacts/motions/walk/motion.npz"
    assert args.upload_wandb is False
    assert args.overwrite is False


def test_parser_accepts_explicit_output_and_opt_in_flags():
    parser = _import_without_simulator_or_wandb().create_parser()

    args = parser.parse_args(
        [
            "--input_file",
            "walk.csv",
            "--output_name",
            "walk",
            "--output_file",
            "custom/motion.npz",
            "--upload_wandb",
            "--overwrite",
        ]
    )

    assert args.output_file == "custom/motion.npz"
    assert args.upload_wandb is True
    assert args.overwrite is True
