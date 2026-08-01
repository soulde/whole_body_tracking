import builtins
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

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


@pytest.mark.parametrize(
    ("headless", "file_saved", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, True, True),
    ],
)
def test_conversion_loop_stops_after_headless_save_but_keeps_interactive_replay(headless, file_saved, expected):
    module = _import_without_simulator_or_wandb()

    assert module._should_continue_conversion(headless=headless, file_saved=file_saved) is expected


def test_require_conversion_output_reports_missing_path(tmp_path):
    module = _import_without_simulator_or_wandb()
    missing = tmp_path / "motion.npz"

    with pytest.raises(RuntimeError, match=f"conversion completed without output: {missing}"):
        module._require_conversion_output(missing)


@pytest.mark.parametrize(
    ("headless", "expected_kwargs"),
    [
        (True, {"wait_for_replicator": False, "skip_cleanup": True}),
        (False, {}),
    ],
)
def test_close_simulation_app_uses_immediate_shutdown_only_in_headless_mode(headless, expected_kwargs):
    module = _import_without_simulator_or_wandb()

    class SimulationApp:
        def __init__(self):
            self.close_kwargs = None

        def close(self, **kwargs):
            self.close_kwargs = kwargs

    app = SimulationApp()
    module._close_simulation_app(app, headless=headless)

    assert app.close_kwargs == expected_kwargs
