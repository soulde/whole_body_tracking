import errno
from pathlib import Path


def resolve_motion_source(motion_file: str | None, registry_name: str | None) -> Path:
    """Resolve a local motion file or download the latest registry artifact."""
    if (motion_file is None) == (registry_name is None):
        raise ValueError("exactly one of motion_file or registry_name is required")

    if motion_file is not None:
        motion_path = Path(motion_file).expanduser().resolve()
        if not motion_path.is_file():
            raise FileNotFoundError(errno.ENOENT, "motion file does not exist", str(motion_path))
        return motion_path

    import wandb

    artifact_name = registry_name if ":" in registry_name else f"{registry_name}:latest"
    artifact = wandb.Api().artifact(artifact_name)
    motion_path = (Path(artifact.download()) / "motion.npz").resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(errno.ENOENT, "artifact does not contain motion.npz", str(motion_path))
    return motion_path
