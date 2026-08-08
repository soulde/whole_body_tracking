"""Pure accumulation and publication utilities for clean teacher evaluation."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike


class EvaluationSummary(NamedTuple):
    episodes: int
    completed: int
    falls: int
    tracking_errors: tuple[float, ...]


class EpisodeAccumulator:
    """Accumulate per-step errors into exactly ``target_episodes`` outcomes."""

    def __init__(self, num_envs: int, target_episodes: int):
        if num_envs < 1 or target_episodes < 1:
            raise ValueError("num_envs and target_episodes must be positive")
        self.num_envs = num_envs
        self.target_episodes = target_episodes
        self._error_sums = np.zeros(num_envs, dtype=np.float64)
        self._step_counts = np.zeros(num_envs, dtype=np.int64)
        self._active = np.ones(num_envs, dtype=bool)
        self._completed = 0
        self._falls = 0
        self._tracking_errors: list[float] = []

    @property
    def done(self) -> bool:
        return len(self._tracking_errors) == self.target_episodes

    def update(self, body_error: ArrayLike, failed: ArrayLike, motion_completed: ArrayLike) -> None:
        errors = np.asarray(body_error, dtype=np.float64)
        failed_mask = np.asarray(failed, dtype=bool)
        completed_mask = np.asarray(motion_completed, dtype=bool)
        expected_shape = (self.num_envs,)
        if errors.shape != expected_shape or failed_mask.shape != expected_shape or completed_mask.shape != expected_shape:
            raise ValueError(f"evaluation arrays must have shape {expected_shape}")
        if not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("body_error must contain finite non-negative values")
        if np.any(failed_mask & completed_mask):
            raise ValueError("failed and motion_completed must be mutually exclusive")
        if self.done:
            return

        self._error_sums[self._active] += errors[self._active]
        self._step_counts[self._active] += 1
        terminal = self._active & (failed_mask | completed_mask)
        remaining = self.target_episodes - len(self._tracking_errors)
        terminal_ids = np.flatnonzero(terminal)[:remaining]
        for env_id in terminal_ids:
            count = int(self._step_counts[env_id])
            if count < 1:
                raise RuntimeError("terminal episode has no tracked steps")
            self._tracking_errors.append(float(self._error_sums[env_id] / count))
            self._completed += int(completed_mask[env_id])
            self._falls += int(failed_mask[env_id])
            self._active[env_id] = False

    def reset(self, env_ids: Sequence[int]) -> None:
        indexes = np.asarray(tuple(env_ids), dtype=np.int64)
        if indexes.size == 0:
            return
        if np.any(indexes < 0) or np.any(indexes >= self.num_envs):
            raise IndexError("environment index out of range")
        self._error_sums[indexes] = 0.0
        self._step_counts[indexes] = 0
        if not self.done:
            self._active[indexes] = True

    def result(self) -> EvaluationSummary:
        if not self.done:
            raise RuntimeError(
                f"evaluation not complete: {len(self._tracking_errors)}/{self.target_episodes} episodes"
            )
        return EvaluationSummary(
            episodes=self.target_episodes,
            completed=self._completed,
            falls=self._falls,
            tracking_errors=tuple(self._tracking_errors),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_evaluation_json(path: Path, payload: Mapping[str, object]) -> str:
    """Publish deterministic JSON atomically without replacing an existing result."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)
