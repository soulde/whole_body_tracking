import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_object:
        for chunk in iter(lambda: file_object.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)


def atomic_save_npz(path: Path, arrays: Mapping[str, NDArray], overwrite: bool = False) -> str:
    """Save arrays to an NPZ file and atomically publish it at ``path``."""
    path = Path(path)
    _require_available(path, overwrite)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".npz",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return sha256_file(path)


def atomic_write_json(path: Path, payload: Mapping[str, object], overwrite: bool = False) -> str:
    """Serialize a mapping deterministically and atomically publish it as JSON."""
    path = Path(path)
    _require_available(path, overwrite)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return sha256_file(path)
