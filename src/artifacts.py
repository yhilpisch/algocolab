"""
Persistent run artifacts shared through Google Drive.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ExperimentConfig


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_id(now: datetime | None = None) -> str:
    """Create a sortable UTC run identifier."""
    current = now or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file to avoid partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class RunBundle:
    """Manage a versioned experiment run directory."""

    path: Path
    manifest: dict[str, Any]

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        config: ExperimentConfig,
        run_id: str | None = None,
        code_commit: str = "unknown",
        data_path: str | Path | None = None,
    ) -> "RunBundle":
        """Create a new run and its initial manifest."""
        config.validate()
        identifier = run_id or make_run_id()
        path = Path(runs_root).expanduser().resolve() / identifier
        path.mkdir(parents=True, exist_ok=False)
        manifest: dict[str, Any] = {
            "schema_version": config.schema_version,
            "run_id": identifier,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit,
            "configuration": config.as_dict(),
            "data_sha256": (
                file_sha256(data_path) if data_path is not None else None
            ),
            "artifacts": {},
            "completed_sessions": [],
            "session_code_commits": {},
        }
        bundle = cls(path=path, manifest=manifest)
        bundle._save_manifest()
        return bundle

    @classmethod
    def open(
        cls,
        runs_root: str | Path,
        run_id: str,
        required_session: int | None = None,
    ) -> "RunBundle":
        """Open and validate an existing run."""
        path = Path(runs_root).expanduser().resolve() / run_id
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Run manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle = cls(path=path, manifest=manifest)
        bundle.validate(required_session=required_session)
        return bundle

    @property
    def run_id(self) -> str:
        """Return the immutable run identifier."""
        return str(self.manifest["run_id"])

    def _save_manifest(self) -> None:
        _write_json(self.path / "manifest.json", self.manifest)

    def _register(self, relative_path: str) -> Path:
        path = self.path / relative_path
        record = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        self.manifest["artifacts"][relative_path] = record
        self._save_manifest()
        return path

    def write_json(self, relative_path: str, payload: Any) -> Path:
        """Write and register a JSON artifact."""
        path = self.path / relative_path
        _write_json(path, payload)
        return self._register(relative_path)

    def write_frame(
        self,
        relative_path: str,
        frame: pd.DataFrame,
        index: bool = False,
    ) -> Path:
        """Write and register a deterministic CSV artifact."""
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=index, lineterminator="\n")
        temporary.replace(path)
        return self._register(relative_path)

    def register_file(self, relative_path: str) -> Path:
        """Register an artifact written by a library such as PyTorch."""
        path = self.path / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return self._register(relative_path)

    def mark_session_complete(
        self,
        session: int,
        required_artifacts: tuple[str, ...],
        code_commit: str | None = None,
    ) -> Path:
        """Validate requirements and create a session completion marker."""
        if session not in (1, 2, 3):
            raise ValueError("Session must be 1, 2, or 3.")
        missing = [
            name
            for name in required_artifacts
            if name not in self.manifest["artifacts"]
        ]
        if missing:
            raise ValueError(f"Unregistered session artifacts: {missing}")
        marker_name = f"SESSION_{session}_COMPLETE"
        marker = self.path / marker_name
        marker.write_text(
            datetime.now(timezone.utc).isoformat() + "\n",
            encoding="utf-8",
        )
        completed = set(self.manifest["completed_sessions"])
        completed.add(session)
        self.manifest["completed_sessions"] = sorted(completed)
        session_commits = self.manifest.setdefault(
            "session_code_commits",
            {},
        )
        session_commits[str(session)] = (
            code_commit or self.manifest["code_commit"]
        )
        self._register(marker_name)
        return marker

    def validate(self, required_session: int | None = None) -> None:
        """Validate manifest identity, completion state, and checksums."""
        if self.manifest.get("run_id") != self.path.name:
            raise ValueError("Manifest run ID does not match its directory.")
        if required_session is not None:
            completed = self.manifest.get("completed_sessions", [])
            if required_session not in completed:
                raise ValueError(
                    f"Session {required_session} is not complete."
                )
        for relative_path, record in self.manifest["artifacts"].items():
            path = self.path / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Artifact not found: {path}")
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {path}")
