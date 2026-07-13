"""Diagnose topology cache files and trained FiLM gamma/beta values.

The script is intentionally independent from ``ppwc.py`` so it can inspect a
checkpoint on a CPU-only machine without importing the PointNet2 CUDA
extension. Only load checkpoints produced by a trusted training run.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


TOPOLOGY_FEATURE_NAMES = (
    "h0_mean_lifetime",
    "h0_max_lifetime",
    "h0_entropy",
    "h1_mean_lifetime",
    "h1_max_lifetime",
    "h1_entropy",
)


def _feature_names(topo_dim: int) -> List[str]:
    if topo_dim == len(TOPOLOGY_FEATURE_NAMES):
        return list(TOPOLOGY_FEATURE_NAMES)
    return [f"feature_{index}" for index in range(topo_dim)]


def _add_finding(report: Dict[str, Any], severity: str, code: str, message: str) -> None:
    report[severity].append({"code": code, "message": message})


def inspect_topology_cache(
    cache_dir: Path,
    topo_dim: int = 6,
    zero_threshold: float = 1e-12,
    max_abs_value: float = 1e6,
    outlier_z: float = 12.0,
    max_examples: int = 5,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Inspect cached topology vectors and return a report plus valid vectors."""
    cache_dir = Path(cache_dir)
    report: Dict[str, Any] = {
        "cache_dir": str(cache_dir.resolve()),
        "file_count": 0,
        "valid_count": 0,
        "errors": [],
        "warnings": [],
        "dimension_stats": [],
    }

    if not cache_dir.is_dir():
        _add_finding(
            report,
            "errors",
            "cache_dir_missing",
            f"Cache directory does not exist: {cache_dir}",
        )
        return report, np.empty((0, topo_dim), dtype=np.float32)

    files = sorted(cache_dir.rglob("*.npy"))
    report["file_count"] = len(files)
    if not files:
        _add_finding(
            report,
            "errors",
            "cache_empty",
            f"No .npy cache files were found below {cache_dir}",
        )
        return report, np.empty((0, topo_dim), dtype=np.float32)

    issue_examples: Dict[str, List[str]] = {
        "load_failed": [],
        "shape_mismatch": [],
        "non_finite": [],
        "negative": [],
        "all_zero": [],
        "magnitude_too_large": [],
        "orphan_temp": [],
    }
    issue_counts = {name: 0 for name in issue_examples}
    valid_vectors: List[np.ndarray] = []
    valid_paths: List[Path] = []

    def record_issue(code: str, path: Path) -> None:
        issue_counts[code] += 1
        if len(issue_examples[code]) < max_examples:
            issue_examples[code].append(str(path))

    for path in files:
        if ".tmp." in path.name:
            record_issue("orphan_temp", path)

        try:
            vector = np.load(path, allow_pickle=False)
        except Exception:
            record_issue("load_failed", path)
            continue

        if vector.shape != (topo_dim,) or not np.issubdtype(vector.dtype, np.number):
            record_issue("shape_mismatch", path)
            continue

        vector = np.asarray(vector, dtype=np.float64)
        if not np.isfinite(vector).all():
            record_issue("non_finite", path)
            continue

        if np.any(vector < -zero_threshold):
            record_issue("negative", path)
        if np.all(np.abs(vector) <= zero_threshold):
            record_issue("all_zero", path)
        if np.max(np.abs(vector)) > max_abs_value:
            record_issue("magnitude_too_large", path)

        valid_vectors.append(vector.astype(np.float32))
        valid_paths.append(path)

    error_messages = {
        "load_failed": "cache files could not be loaded",
        "shape_mismatch": f"cache files were not numeric vectors with shape ({topo_dim},)",
        "non_finite": "cache files contained NaN or infinity",
        "negative": "cache files contained negative lifetime/entropy values",
        "all_zero": "cache files were all zero (the extractor fallback value)",
        "magnitude_too_large": f"cache files exceeded max absolute value {max_abs_value:g}",
    }
    for code, message in error_messages.items():
        count = issue_counts[code]
        if count:
            examples = "; ".join(issue_examples[code])
            _add_finding(
                report,
                "errors",
                code,
                f"{count} {message}. Examples: {examples}",
            )

    if issue_counts["orphan_temp"]:
        examples = "; ".join(issue_examples["orphan_temp"])
        _add_finding(
            report,
            "warnings",
            "orphan_temp",
            f"{issue_counts['orphan_temp']} possible interrupted-write temp files. "
            f"Examples: {examples}",
        )

    if not valid_vectors:
        _add_finding(
            report,
            "errors",
            "no_valid_vectors",
            "No structurally valid, finite topology vectors are available for statistics",
        )
        return report, np.empty((0, topo_dim), dtype=np.float32)

    matrix = np.stack(valid_vectors, axis=0)
    report["valid_count"] = int(matrix.shape[0])
    names = _feature_names(topo_dim)
    for index, name in enumerate(names):
        values = matrix[:, index].astype(np.float64)
        report["dimension_stats"].append(
            {
                "name": name,
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }
        )

    if matrix.shape[0] >= 2:
        constant_dims = [
            names[index]
            for index in range(topo_dim)
            if np.ptp(matrix[:, index]) <= zero_threshold
        ]
        if constant_dims:
            _add_finding(
                report,
                "warnings",
                "constant_dimensions",
                "No variation was observed in: " + ", ".join(constant_dims),
            )

    if matrix.shape[0] >= 8 and outlier_z > 0:
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        active_dims = mad > zero_threshold
        robust_z = np.zeros_like(matrix, dtype=np.float64)
        robust_z[:, active_dims] = (
            0.6744897501960817
            * np.abs(matrix[:, active_dims] - median[active_dims])
            / mad[active_dims]
        )
        outlier_rows = np.flatnonzero(np.any(robust_z > outlier_z, axis=1))
        if outlier_rows.size:
            examples = "; ".join(
                f"{valid_paths[index]} (robust_z={np.max(robust_z[index]):.2f})"
                for index in outlier_rows[:max_examples]
            )
            _add_finding(
                report,
                "warnings",
                "statistical_outliers",
                f"{outlier_rows.size} cache vectors are possible robust outliers. "
                f"Examples: {examples}",
            )

    return report, matrix


def _load_state_dict(checkpoint_path: Path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint inspection") from exc

    load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    checkpoint = torch.load(str(checkpoint_path), **load_kwargs)

    preferred_keys = ("state_dict", "model_state_dict", "model", "net", "network")

    def find_state_dict(candidate):
        if hasattr(candidate, "state_dict") and not isinstance(candidate, Mapping):
            return candidate.state_dict()
        if not isinstance(candidate, Mapping):
            return None
        if any(re.search(r"(?:^|\.)(?:gamma_gen|beta_gen)\.(?:weight|bias)$", str(key)) for key in candidate):
            return candidate
        for key in preferred_keys:
            if key in candidate:
                nested = find_state_dict(candidate[key])
                if nested is not None:
                    return nested
        return None

    state_dict = find_state_dict(checkpoint)
    if state_dict is None:
        raise RuntimeError(
            "Could not find gamma_gen/beta_gen tensors in the checkpoint or its common state-dict fields"
        )
    return state_dict


def _numeric_stats(values: np.ndarray, near_zero_threshold: float) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_values = np.abs(values)
    return {
        "numel": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(abs_values)),
        "max_abs": float(np.max(abs_values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "fraction_near_zero": float(np.mean(abs_values <= near_zero_threshold)),
        "near_zero": bool(np.max(abs_values) <= near_zero_threshold),
    }


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def _evaluate_film_outputs(
    state_dict,
    topology_vectors: np.ndarray,
    near_zero_threshold: float,
    max_samples: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Replay the repository's Linear -> LayerNorm -> LeakyReLU FiLM heads."""
    import torch
    import torch.nn.functional as functional

    output_reports: List[Dict[str, Any]] = []
    warnings: List[str] = []
    gamma_weight_pattern = re.compile(r"^(.*)gamma_gen\.weight$")

    for key in sorted(str(name) for name in state_dict):
        match = gamma_weight_pattern.match(key)
        if not match:
            continue
        prefix = match.group(1)
        required = {
            "linear_weight": f"{prefix}film_gen.0.weight",
            "linear_bias": f"{prefix}film_gen.0.bias",
            "norm_weight": f"{prefix}film_gen.1.weight",
            "norm_bias": f"{prefix}film_gen.1.bias",
            "gamma_weight": f"{prefix}gamma_gen.weight",
            "gamma_bias": f"{prefix}gamma_gen.bias",
            "beta_weight": f"{prefix}beta_gen.weight",
            "beta_bias": f"{prefix}beta_gen.bias",
        }
        missing = [name for name in required.values() if name not in state_dict]
        module_name = prefix[:-1] if prefix.endswith(".") else prefix or "<root>"
        if missing:
            warnings.append(
                f"Cannot replay {module_name}; missing tensors: {', '.join(missing)}"
            )
            continue

        input_dim = int(state_dict[required["linear_weight"]].shape[1])
        if topology_vectors.ndim != 2 or topology_vectors.shape[1] != input_dim:
            warnings.append(
                f"Cannot replay {module_name}; cache dimension is "
                f"{topology_vectors.shape[1] if topology_vectors.ndim == 2 else 'invalid'}, "
                f"but film_gen expects {input_dim}"
            )
            continue

        samples = torch.from_numpy(topology_vectors[:max_samples]).float()
        with torch.no_grad():
            hidden = functional.linear(
                samples,
                state_dict[required["linear_weight"]].detach().cpu().float(),
                state_dict[required["linear_bias"]].detach().cpu().float(),
            )
            hidden = functional.layer_norm(
                hidden,
                (hidden.shape[-1],),
                state_dict[required["norm_weight"]].detach().cpu().float(),
                state_dict[required["norm_bias"]].detach().cpu().float(),
                eps=1e-5,
            )
            hidden = functional.leaky_relu(hidden, negative_slope=0.1)
            gamma = functional.linear(
                hidden,
                state_dict[required["gamma_weight"]].detach().cpu().float(),
                state_dict[required["gamma_bias"]].detach().cpu().float(),
            )
            beta = functional.linear(
                hidden,
                state_dict[required["beta_weight"]].detach().cpu().float(),
                state_dict[required["beta_bias"]].detach().cpu().float(),
            )

        output_reports.append(
            {
                "module": module_name,
                "sample_count": int(samples.shape[0]),
                "gamma": _numeric_stats(_to_numpy(gamma), near_zero_threshold),
                "beta": _numeric_stats(_to_numpy(beta), near_zero_threshold),
            }
        )

    return output_reports, warnings


def inspect_film_checkpoint(
    checkpoint_path: Path,
    topology_vectors: np.ndarray,
    near_zero_threshold: float = 1e-6,
    max_samples: int = 10000,
) -> Dict[str, Any]:
    """Inspect FiLM generator parameters and, when possible, their outputs."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    state_dict = _load_state_dict(checkpoint_path)
    report: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "near_zero_threshold": near_zero_threshold,
        "parameter_groups": [],
        "output_groups": [],
        "errors": [],
        "warnings": [],
    }
    head_pattern = re.compile(r"(?:^|\.)(gamma_gen|beta_gen)\.(weight|bias)$")
    grouped_values: Dict[str, List[np.ndarray]] = {"gamma": [], "beta": []}
    grouped_names: Dict[str, List[str]] = {"gamma": [], "beta": []}

    for name, tensor in state_dict.items():
        match = head_pattern.search(str(name))
        if not match or not hasattr(tensor, "detach"):
            continue
        group = "gamma" if match.group(1) == "gamma_gen" else "beta"
        grouped_values[group].append(_to_numpy(tensor).reshape(-1))
        grouped_names[group].append(str(name))

    for group in ("gamma", "beta"):
        if not grouped_values[group]:
            _add_finding(
                report,
                "errors",
                f"{group}_parameters_missing",
                f"No {group}_gen weight or bias tensors were found",
            )
            continue
        values = np.concatenate(grouped_values[group])
        report["parameter_groups"].append(
            {
                "name": f"{group}_gen parameters",
                "tensor_names": grouped_names[group],
                "stats": _numeric_stats(values, near_zero_threshold),
            }
        )

    if topology_vectors.size:
        outputs, replay_warnings = _evaluate_film_outputs(
            state_dict,
            topology_vectors,
            near_zero_threshold,
            max_samples,
        )
        report["output_groups"] = outputs
        for message in replay_warnings:
            _add_finding(report, "warnings", "film_replay_skipped", message)
        if not outputs:
            _add_finding(
                report,
                "warnings",
                "no_film_outputs",
                "FiLM outputs could not be replayed from this checkpoint layout",
            )
    else:
        _add_finding(
            report,
            "warnings",
            "no_topology_samples",
            "No valid topology vectors were available, so only parameters were checked",
        )

    return report


def _print_findings(title: str, findings: Sequence[Dict[str, str]]) -> None:
    if not findings:
        return
    print(title)
    for finding in findings:
        print(f"  - [{finding['code']}] {finding['message']}")


def print_cache_report(report: Dict[str, Any]) -> None:
    print("\n=== Topology cache ===")
    print(f"Directory: {report['cache_dir']}")
    print(f"Files: {report['file_count']}  Valid numeric vectors: {report['valid_count']}")
    if report["dimension_stats"]:
        print("Per-dimension distribution:")
        for stats in report["dimension_stats"]:
            print(
                f"  {stats['name']:<18} min={stats['min']:.6g} "
                f"median={stats['median']:.6g} mean={stats['mean']:.6g} "
                f"max={stats['max']:.6g} std={stats['std']:.6g}"
            )
    _print_findings("Errors:", report["errors"])
    _print_findings("Warnings:", report["warnings"])


def _format_stats(stats: Dict[str, Any]) -> str:
    return (
        f"numel={stats['numel']} mean={stats['mean']:.6g} "
        f"mean_abs={stats['mean_abs']:.6g} max_abs={stats['max_abs']:.6g} "
        f"rms={stats['rms']:.6g} near_zero_fraction={stats['fraction_near_zero']:.2%} "
        f"near_zero={stats['near_zero']}"
    )


def print_checkpoint_report(report: Dict[str, Any]) -> None:
    print("\n=== FiLM gamma/beta ===")
    print(f"Checkpoint: {report['checkpoint']}")
    print(f"Near-zero threshold: {report['near_zero_threshold']:.6g}")
    print("Generator parameters:")
    for group in report["parameter_groups"]:
        print(f"  {group['name']}: {_format_stats(group['stats'])}")
        for name in group["tensor_names"]:
            print(f"    - {name}")
    if report["output_groups"]:
        print("Outputs replayed from cached topology vectors:")
        for group in report["output_groups"]:
            print(f"  {group['module']} ({group['sample_count']} samples)")
            print(f"    gamma: {_format_stats(group['gamma'])}")
            print(f"    beta:  {_format_stats(group['beta'])}")
    _print_findings("Errors:", report["errors"])
    _print_findings("Warnings:", report["warnings"])


def _has_near_zero_film(report: Dict[str, Any]) -> bool:
    if any(group["stats"]["near_zero"] for group in report["parameter_groups"]):
        return True
    return any(
        group[head]["near_zero"]
        for group in report["output_groups"]
        for head in ("gamma", "beta")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check topology .npy cache integrity and whether trained FiLM "
            "gamma/beta generators remain close to zero."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("topo_cache"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--topo-dim", type=int, default=6)
    parser.add_argument(
        "--cache-zero-threshold",
        type=float,
        default=1e-12,
        help="absolute tolerance used to classify an entire cache vector as zero",
    )
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=1e-6,
        help="max absolute value at or below which a gamma/beta group is near zero",
    )
    parser.add_argument("--max-cache-abs", type=float, default=1e6)
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=12.0,
        help="robust modified-z warning threshold; set to 0 to disable",
    )
    parser.add_argument("--max-film-samples", type=int, default=10000)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also return exit code 1 for statistical/cache warnings",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.topo_dim <= 0:
        parser.error("--topo-dim must be positive")
    if args.cache_zero_threshold < 0 or args.near_zero_threshold < 0:
        parser.error("zero thresholds must be non-negative")
    if args.max_cache_abs <= 0:
        parser.error("--max-cache-abs must be positive")
    if args.outlier_z < 0:
        parser.error("--outlier-z must be non-negative")
    if args.max_film_samples <= 0 or args.max_examples <= 0:
        parser.error("sample/example limits must be positive")

    cache_report, topology_vectors = inspect_topology_cache(
        args.cache_dir,
        topo_dim=args.topo_dim,
        zero_threshold=args.cache_zero_threshold,
        max_abs_value=args.max_cache_abs,
        outlier_z=args.outlier_z,
        max_examples=args.max_examples,
    )
    print_cache_report(cache_report)

    try:
        checkpoint_report = inspect_film_checkpoint(
            args.checkpoint,
            topology_vectors,
            near_zero_threshold=args.near_zero_threshold,
            max_samples=args.max_film_samples,
        )
    except Exception as exc:
        print(f"\n[FATAL] Checkpoint inspection failed: {exc}", file=sys.stderr)
        return 2

    print_checkpoint_report(checkpoint_report)
    cache_abnormal = bool(cache_report["errors"])
    checkpoint_abnormal = bool(checkpoint_report["errors"])
    near_zero = _has_near_zero_film(checkpoint_report)
    warned = bool(cache_report["warnings"] or checkpoint_report["warnings"])

    print("\n=== Verdict ===")
    print(f"Cache abnormal: {cache_abnormal}")
    print(f"Checkpoint structure abnormal: {checkpoint_abnormal}")
    print(f"Gamma/beta near zero: {near_zero}")
    if cache_abnormal or checkpoint_abnormal or near_zero or (args.strict and warned):
        print("Result: ATTENTION (exit code 1)")
        return 1
    print("Result: OK (exit code 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
