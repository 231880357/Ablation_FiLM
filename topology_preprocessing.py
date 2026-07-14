"""Shared topology-cache identity and normalization utilities.

Raw topology vectors stay in ``.npy`` cache files.  Z-score normalization is
applied only when a vector is passed to the model, using statistics generated
from the training cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
TOPOLOGY_STATS_SCHEMA_VERSION = 1
_CACHE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_project_path(path_value: str | os.PathLike[str]) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_cache_version(cache_version: str) -> str:
    """Return a path-safe cache version or raise a clear configuration error."""
    value = str(cache_version).strip()
    if not _CACHE_VERSION_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "DATA.TOPO_CACHE_VERSION must contain only letters, numbers, '.', "
            "'_' or '-', and must start with a letter or number"
        )
    return value


def point_cloud_fingerprint(point_cloud: np.ndarray) -> str:
    """Hash canonical float32 point content, including its shape."""
    array = np.ascontiguousarray(np.asarray(point_cloud, dtype=np.float32))
    if not np.isfinite(array).all():
        raise ValueError("Cannot cache topology for a point cloud containing NaN/Inf")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def topology_cache_record_id(
    cache_key: str,
    point_cloud: np.ndarray,
    topo_dim: int,
    cache_version: str,
) -> str:
    """Build a content-addressed id for one raw topology vector."""
    version = validate_cache_version(cache_version)
    payload = {
        "cache_key": str(cache_key),
        "cache_version": version,
        "point_cloud_sha256": point_cloud_fingerprint(point_cloud),
        "topo_dim": int(topo_dim),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_topology_vector(vector: np.ndarray, topo_dim: int) -> np.ndarray:
    """Validate and return one raw topology vector as float32."""
    array = np.asarray(vector)
    if array.shape != (int(topo_dim),):
        raise ValueError(f"Expected topology shape ({topo_dim},), got {array.shape}")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"Expected numeric topology dtype, got {array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Topology vector contains NaN/Inf")
    return array


def load_topology_cache_file(path: Path, topo_dim: int) -> np.ndarray:
    """Load one cache file without pickle and validate its content."""
    return validate_topology_vector(np.load(path, allow_pickle=False), topo_dim)


def topology_vector_digest(vector: np.ndarray, topo_dim: int) -> str:
    """Hash the canonical raw float32 bytes of one topology vector."""
    array = validate_topology_vector(vector, topo_dim)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _unique_cache_files(cache_files: Iterable[Path]) -> List[Path]:
    files = sorted({Path(path).resolve() for path in cache_files}, key=lambda item: str(item))
    if not files:
        raise ValueError("No topology cache files were provided")
    return files


def cache_files_digest(
    cache_files: Sequence[Path],
    source_cache_dir: Path,
    topo_dim: int,
) -> str:
    """Hash active cache filenames and canonical raw vectors deterministically."""
    source_cache_dir = Path(source_cache_dir).resolve()
    digest = hashlib.sha256()
    for path in _unique_cache_files(cache_files):
        try:
            relative = path.relative_to(source_cache_dir)
        except ValueError as exc:
            raise ValueError(f"Cache file is outside source directory: {path}") from exc
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        vector = load_topology_cache_file(path, topo_dim)
        digest.update(topology_vector_digest(vector, topo_dim).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _portable_source_directory(source_cache_dir: Path) -> Dict[str, Any]:
    source_cache_dir = Path(source_cache_dir).resolve()
    try:
        relative = source_cache_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        return {"path": str(source_cache_dir), "relative_to_project": False}
    return {"path": relative.as_posix(), "relative_to_project": True}


def _resolve_stats_source_directory(stats: Dict[str, Any]) -> Path:
    source = stats.get("source_cache_dir")
    if not isinstance(source, dict) or "path" not in source:
        raise ValueError("Topology stats are missing source_cache_dir metadata")
    path = Path(str(source["path"]))
    if bool(source.get("relative_to_project")):
        path = PROJECT_ROOT / path
    return path.resolve()


def write_topology_stats(
    cache_files: Sequence[Path],
    source_cache_dir: Path,
    output_path: Path,
    topo_dim: int,
    cache_version: str,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Write train-only z-score statistics and cache provenance as JSON."""
    version = validate_cache_version(cache_version)
    if topo_dim <= 0:
        raise ValueError("topo_dim must be positive")
    if eps <= 0:
        raise ValueError("normalization epsilon must be positive")

    source_cache_dir = Path(source_cache_dir).resolve()
    files = _unique_cache_files(cache_files)
    vectors = [load_topology_cache_file(path, topo_dim) for path in files]
    matrix = np.stack(vectors, axis=0).astype(np.float64)
    all_zero_rows = np.flatnonzero(np.all(np.abs(matrix) <= eps, axis=1))
    if all_zero_rows.size:
        raise ValueError(
            f"Cannot generate stats from {all_zero_rows.size} all-zero topology vectors"
        )
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    small_dims = np.flatnonzero(std <= eps)
    if small_dims.size:
        raise ValueError(
            "Cannot z-score constant topology dimensions: "
            + ", ".join(str(int(index)) for index in small_dims)
        )

    cache_entries = []
    for path in files:
        try:
            relative = path.relative_to(source_cache_dir).as_posix()
        except ValueError as exc:
            raise ValueError(f"Cache file is outside source directory: {path}") from exc
        cache_entries.append(
            {
                "file": relative,
                "raw_sha256": topology_vector_digest(
                    load_topology_cache_file(path, topo_dim), topo_dim
                ),
            }
        )

    stats: Dict[str, Any] = {
        "schema_version": TOPOLOGY_STATS_SCHEMA_VERSION,
        "cache_version": version,
        "topo_dim": int(topo_dim),
        "sample_count": int(matrix.shape[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "normalization": "zscore",
        "normalization_eps": float(eps),
        "source_cache_dir": _portable_source_directory(source_cache_dir),
        "cache_entries": cache_entries,
        "cache_digest_sha256": cache_files_digest(
            files,
            source_cache_dir,
            topo_dim,
        ),
    }

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(stats, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return stats


def load_topology_stats(stats_path: Path) -> Dict[str, Any]:
    """Load and structurally validate a topology statistics JSON file."""
    path = Path(stats_path)
    if not path.is_absolute():
        path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Topology normalization stats do not exist: {path}. "
            "Run prepare_topology_cache.py first."
        )
    with path.open("r", encoding="utf-8") as stream:
        stats = json.load(stream)
    if not isinstance(stats, dict):
        raise ValueError(f"Topology stats must be a JSON object: {path}")
    if stats.get("schema_version") != TOPOLOGY_STATS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported topology stats schema in {path}: "
            f"{stats.get('schema_version')!r}"
        )
    return stats


@dataclass
class TopologyNormalizer:
    """Apply a validated topology normalization policy."""

    mode: str
    topo_dim: int
    cache_version: str
    eps: float
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    stats_path: Path | None = None
    stats: Dict[str, Any] | None = None
    entry_digests: Dict[str, str] | None = None

    @classmethod
    def from_stats_file(
        cls,
        stats_path: Path,
        topo_dim: int,
        expected_cache_version: str | None = None,
        eps: float = 1e-6,
    ) -> "TopologyNormalizer":
        path = Path(stats_path)
        if not path.is_absolute():
            path = resolve_project_path(path)
        stats = load_topology_stats(path)
        if stats.get("normalization") != "zscore":
            raise ValueError(f"Topology stats are not z-score statistics: {path}")
        if int(stats.get("topo_dim", -1)) != int(topo_dim):
            raise ValueError(
                f"Topology stats dim {stats.get('topo_dim')} does not match model dim {topo_dim}"
            )
        cache_version = validate_cache_version(str(stats.get("cache_version", "")))
        if expected_cache_version is not None:
            expected = validate_cache_version(expected_cache_version)
            if cache_version != expected:
                raise ValueError(
                    f"Topology stats cache version {cache_version!r} "
                    f"does not match expected version {expected!r}"
                )
        if eps <= 0:
            raise ValueError("normalization epsilon must be positive")
        if int(stats.get("sample_count", 0)) <= 0:
            raise ValueError("Topology stats sample_count must be positive")

        mean = np.asarray(stats.get("mean"), dtype=np.float32)
        std = np.asarray(stats.get("std"), dtype=np.float32)
        if mean.shape != (topo_dim,) or std.shape != (topo_dim,):
            raise ValueError(
                f"Topology stats mean/std must both have shape ({topo_dim},)"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Topology stats contain NaN/Inf")
        if np.any(std <= eps):
            raise ValueError("Topology stats contain a zero or near-zero standard deviation")

        entry_digests = {
            str(entry["file"]): str(entry["raw_sha256"])
            for entry in stats.get("cache_entries", [])
            if isinstance(entry, dict) and "file" in entry and "raw_sha256" in entry
        }
        if len(entry_digests) != int(stats["sample_count"]):
            raise ValueError("Topology stats cache_entries do not match sample_count")
        return cls(
            mode="zscore",
            topo_dim=int(topo_dim),
            cache_version=cache_version,
            eps=float(eps),
            mean=mean,
            std=std,
            stats_path=path,
            stats=stats,
            entry_digests=entry_digests,
        )

    @classmethod
    def from_config(
        cls,
        cfg,
        topo_dim: int,
        verify_cache_digest: bool = False,
    ) -> "TopologyNormalizer":
        mode = str(getattr(cfg.DATA, "TOPO_NORMALIZATION", "none")).strip().lower()
        cache_version = validate_cache_version(
            getattr(cfg.DATA, "TOPO_CACHE_VERSION", "v2")
        )
        eps = float(getattr(cfg.DATA, "TOPO_NORM_EPS", 1e-6))
        if mode not in {"none", "zscore"}:
            raise ValueError(
                "DATA.TOPO_NORMALIZATION must be 'none' or 'zscore', "
                f"got {mode!r}"
            )
        if eps <= 0:
            raise ValueError("DATA.TOPO_NORM_EPS must be positive")
        if topo_dim <= 0 or mode == "none":
            return cls(mode="none", topo_dim=int(topo_dim), cache_version=cache_version, eps=eps)

        stats_value = str(getattr(cfg.DATA, "TOPO_STATS_PATH", "")).strip()
        if not stats_value:
            raise ValueError(
                "DATA.TOPO_STATS_PATH is required when TOPO_NORMALIZATION='zscore'"
            )
        normalizer = cls.from_stats_file(
            resolve_project_path(stats_value),
            topo_dim,
            expected_cache_version=cache_version,
            eps=eps,
        )
        if verify_cache_digest:
            normalizer.verify_source_cache()
        return normalizer

    @property
    def enabled(self) -> bool:
        return self.mode == "zscore"

    @property
    def source_cache_directory(self) -> Path | None:
        if not self.enabled or self.stats is None:
            return None
        return _resolve_stats_source_directory(self.stats)

    def manifest_cache_files(self, source_cache_dir: Path | None = None) -> List[Path]:
        """Resolve the ordered cache files represented by the stats manifest."""
        if not self.enabled or self.stats is None:
            return []
        directory = (
            self.source_cache_directory
            if source_cache_dir is None
            else Path(source_cache_dir).resolve()
        )
        if directory is None:
            return []
        entries = self.stats.get("cache_entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Topology stats do not list their source cache entries")
        return [directory / str(entry["file"]) for entry in entries]

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        if self.topo_dim <= 0 and not self.enabled:
            return np.asarray(vector, dtype=np.float32).copy()
        array = validate_topology_vector(vector, self.topo_dim)
        if not self.enabled:
            return array.copy()
        return np.asarray((array - self.mean) / self.std, dtype=np.float32)

    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.topo_dim:
            raise ValueError(
                f"Expected topology matrix [N, {self.topo_dim}], got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Topology matrix contains NaN/Inf")
        if not self.enabled:
            return array.copy()
        return np.asarray((array - self.mean[None, :]) / self.std[None, :], dtype=np.float32)

    def verify_source_cache(self) -> None:
        if not self.enabled or self.stats is None:
            return
        source_cache_dir = self.source_cache_directory
        self.verify_cache_directory(source_cache_dir)

    def verify_cache_directory(self, source_cache_dir: Path) -> None:
        """Verify the complete stats manifest in a concrete cache directory."""
        if not self.enabled or self.stats is None:
            return
        source_cache_dir = Path(source_cache_dir).resolve()
        files = self.manifest_cache_files(source_cache_dir)
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Topology stats source cache is incomplete. Missing: "
                + ", ".join(missing[:5])
            )
        actual_digest = cache_files_digest(files, source_cache_dir, self.topo_dim)
        expected_digest = str(self.stats.get("cache_digest_sha256", ""))
        if not expected_digest or actual_digest != expected_digest:
            raise ValueError(
                "Topology training-cache digest does not match the normalization stats. "
                "Run prepare_topology_cache.py again."
            )

    def verify_cache_entry(
        self,
        cache_path: Path,
        source_cache_dir: Path,
        raw_vector: np.ndarray,
    ) -> None:
        """Verify one rebuilt/loaded train entry against the stats manifest."""
        if not self.enabled:
            return
        cache_path = Path(cache_path).resolve()
        source_cache_dir = Path(source_cache_dir).resolve()
        try:
            relative = cache_path.relative_to(source_cache_dir).as_posix()
        except ValueError as exc:
            raise ValueError(f"Topology cache entry is outside its versioned directory: {cache_path}") from exc
        expected = (self.entry_digests or {}).get(relative)
        if expected is None:
            raise ValueError(
                f"Topology cache identity {relative} is not represented by {self.stats_path}. "
                "The point cloud or preprocessing changed; run prepare_topology_cache.py again."
            )
        actual = topology_vector_digest(raw_vector, self.topo_dim)
        if actual != expected:
            raise ValueError(
                f"Topology cache value {relative} does not match {self.stats_path}. "
                "Run prepare_topology_cache.py again."
            )
