"""Consistent input snapshots and create-only publication for local artifacts."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self


def file_sha256(path: Path) -> str:
    """Hash on-disk bytes without retaining a large model file in memory."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@dataclass(frozen=True)
class FileSnapshot:
    """Use the same immutable bytes for parsing and provenance, even if the path changes."""

    path: Path
    payload: bytes

    @classmethod
    def read(cls, path: Path) -> Self:
        """Capture a source exactly once; consumers must not reopen it to parse or hash."""
        path = Path(path)
        return cls(path, path.read_bytes())

    @property
    def sha256(self) -> str:
        """Identify the captured content, rather than whatever now occupies the original path."""
        return hashlib.sha256(self.payload).hexdigest()

    def assert_unchanged(self) -> None:
        """Refuse to publish a training artifact after its source files have changed."""
        if file_sha256(self.path) != self.sha256:
            raise ValueError(
                f"dataset changed during training: {self.path.name}; no head published"
            )


def write_json(path: Path, value: object) -> None:
    """Publish complete JSON atomically and exclusively; never truncate an existing version."""
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        # Same-directory hard linking provides atomic create-if-absent on Mac/Linux.
        # replace() would silently overwrite a concurrently published report or manifest.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
