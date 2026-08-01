import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "motion_source.py"
)
SPEC = importlib.util.spec_from_file_location("motion_source_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOTION_SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOTION_SOURCE)
resolve_motion_source = MOTION_SOURCE.resolve_motion_source


@pytest.mark.parametrize(
    ("motion_file", "registry_name"),
    [
        (None, None),
        ("motion.npz", "team/project/motion"),
    ],
)
def test_resolve_motion_source_requires_exactly_one_source(motion_file, registry_name):
    with pytest.raises(ValueError):
        resolve_motion_source(motion_file, registry_name)


def test_resolve_motion_source_rejects_missing_local_file(tmp_path):
    missing = tmp_path / "missing.npz"

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_motion_source(str(missing), None)

    assert exc_info.value.filename == str(missing.resolve())


def test_resolve_motion_source_resolves_local_path_without_importing_wandb(tmp_path):
    motion = tmp_path / "motion.npz"
    motion.touch()

    with patch.dict(sys.modules, {"wandb": None}):
        assert resolve_motion_source(str(motion), None) == motion.resolve()


def test_resolve_motion_source_downloads_latest_registry_artifact(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    motion = downloaded / "motion.npz"
    motion.touch()
    requested_names = []

    class FakeArtifact:
        def download(self):
            return str(downloaded)

    class FakeApi:
        def artifact(self, name):
            requested_names.append(name)
            return FakeArtifact()

    fake_wandb = SimpleNamespace(Api=FakeApi)

    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        result = resolve_motion_source(None, "team/project/walk")

    assert result == motion.resolve()
    assert requested_names == ["team/project/walk:latest"]


def test_resolve_motion_source_requires_motion_npz_in_registry_artifact(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()

    class FakeArtifact:
        def download(self):
            return str(downloaded)

    class FakeApi:
        def artifact(self, name):
            return FakeArtifact()

    fake_wandb = SimpleNamespace(Api=FakeApi)

    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_motion_source(None, "team/project/walk")

    assert exc_info.value.filename == str((downloaded / "motion.npz").resolve())
