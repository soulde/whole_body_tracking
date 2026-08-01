import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

RUNNER_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "my_on_policy_runner.py"
)


class _FakeOnPolicyRunner:
    def save(self, path, infos=None):
        del path, infos


def _import_runner_without_wandb():
    fake_modules = {
        "wandb": None,
        "rsl_rl.env": SimpleNamespace(VecEnv=object),
        "rsl_rl.runners.on_policy_runner": SimpleNamespace(OnPolicyRunner=_FakeOnPolicyRunner),
        "isaaclab_rl.rsl_rl": SimpleNamespace(export_policy_as_onnx=lambda *args, **kwargs: None),
        "whole_body_tracking.utils.exporter": SimpleNamespace(
            attach_onnx_metadata=lambda *args, **kwargs: None,
            export_motion_policy_as_onnx=lambda *args, **kwargs: None,
        ),
    }
    spec = importlib.util.spec_from_file_location("runner_under_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


def test_tensorboard_motion_runner_saves_checkpoint_and_local_onnx_without_wandb(tmp_path, monkeypatch):
    module = _import_runner_without_wandb()
    checkpoint = tmp_path / "run" / "model_1.pt"

    def save_checkpoint(_runner, path, infos=None):
        del infos
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"checkpoint")

    def export_onnx(_env, _policy, normalizer, path, filename):
        assert normalizer is runner.obs_normalizer
        output = Path(path) / filename
        output.write_bytes(b"onnx")

    monkeypatch.setattr(module.OnPolicyRunner, "save", save_checkpoint)
    monkeypatch.setattr(module, "export_motion_policy_as_onnx", export_onnx)
    monkeypatch.setattr(module, "attach_onnx_metadata", lambda *args, **kwargs: None)

    runner = object.__new__(module.MotionOnPolicyRunner)
    runner.logger_type = "tensorboard"
    runner.env = SimpleNamespace(unwrapped=object())
    runner.alg = SimpleNamespace(policy=object())
    runner.obs_normalizer = object()
    runner.registry_name = None

    with patch.dict(sys.modules, {"wandb": None}):
        runner.save(str(checkpoint))

    assert checkpoint.read_bytes() == b"checkpoint"
    assert (checkpoint.parent / "run.onnx").read_bytes() == b"onnx"


def test_wandb_motion_runner_skips_remote_calls_without_a_live_run(tmp_path, monkeypatch):
    module = _import_runner_without_wandb()
    checkpoint = tmp_path / "run" / "model_1.pt"

    def save_checkpoint(_runner, path, infos=None):
        del infos
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"checkpoint")

    def export_onnx(_env, _policy, normalizer, path, filename):
        del normalizer
        (Path(path) / filename).write_bytes(b"onnx")

    fake_wandb = SimpleNamespace(
        run=None,
        save=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wandb.save must require a live run")),
    )
    monkeypatch.setattr(module.OnPolicyRunner, "save", save_checkpoint)
    monkeypatch.setattr(module, "export_motion_policy_as_onnx", export_onnx)
    monkeypatch.setattr(module, "attach_onnx_metadata", lambda *args, **kwargs: None)

    runner = object.__new__(module.MotionOnPolicyRunner)
    runner.logger_type = "wandb"
    runner.env = SimpleNamespace(unwrapped=object())
    runner.alg = SimpleNamespace(policy=object())
    runner.obs_normalizer = object()
    runner.registry_name = "team/project/walk"

    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        runner.save(str(checkpoint))

    assert checkpoint.read_bytes() == b"checkpoint"
    assert (checkpoint.parent / "run.onnx").read_bytes() == b"onnx"
    assert runner.registry_name == "team/project/walk"
