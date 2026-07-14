"""Prepare versioned Lung topology caches and train-only z-score statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from defaults import get_cfg_defaults
from topology_preprocessing import resolve_project_path, write_topology_stats


def _warm_cache(dataset, label: str) -> None:
    total = len(dataset)
    for index in range(total):
        dataset[index]
        completed = index + 1
        if completed == total or completed % 10 == 0:
            print(f"{label}: cached {completed}/{total} cases")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate raw, content-addressed Lung topology caches and compute "
            "z-score statistics from the training split only."
        )
    )
    parser.add_argument("--config", default="config_ppwc_sup_toponorm.yaml")
    parser.add_argument(
        "--cloudfolder-train",
        "--cloudfolder_train",
        dest="cloudfolder_train",
        required=True,
    )
    parser.add_argument(
        "--cloudfolder-val",
        "--cloudfolder_val",
        dest="cloudfolder_val",
        required=True,
    )
    parser.add_argument(
        "--supfolder-train",
        "--supfolder_train",
        dest="supfolder_train",
        required=True,
    )
    parser.add_argument(
        "--supfolder-val",
        "--supfolder_val",
        dest="supfolder_val",
        required=True,
    )
    parser.add_argument("--lung-index-file", required=True)
    parser.add_argument(
        "--stats-out",
        default=None,
        help="override DATA.TOPO_STATS_PATH from the config",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="prepare only train cache and statistics",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    if int(cfg.MODEL.TOPO_FEAT_DIM) <= 0:
        parser.error("MODEL.TOPO_FEAT_DIM must be positive")
    if not bool(cfg.DATA.TOPO_CACHE_ENABLED):
        parser.error("DATA.TOPO_CACHE_ENABLED must be true")
    if str(cfg.DATA.TOPO_NORMALIZATION).lower() != "zscore":
        parser.error("the preparation config must set DATA.TOPO_NORMALIZATION='zscore'")

    args.prepare_topology_cache = True
    # Imported after argument/config validation so --help does not require Torch.
    from dataset import Lung250MDataset

    train_set = Lung250MDataset(cfg, args, phase="cache", split="train")
    print(f"Preparing raw train topology cache: {train_set.topo_cache_dir}")
    _warm_cache(train_set, "train")
    expected_train_files = len(train_set) * 2
    if len(train_set.topo_cache_files_used) != expected_train_files:
        raise RuntimeError(
            f"Expected {expected_train_files} active train cache files, got "
            f"{len(train_set.topo_cache_files_used)}"
        )

    train_cache_files = [Path(path) for path in train_set.topo_cache_files_used]

    if not args.skip_val:
        val_set = Lung250MDataset(cfg, args, phase="cache", split="val")
        print(f"Preparing raw validation topology cache: {val_set.topo_cache_dir}")
        _warm_cache(val_set, "val")
        expected_val_files = len(val_set) * 2
        if len(val_set.topo_cache_files_used) != expected_val_files:
            raise RuntimeError(
                f"Expected {expected_val_files} active validation cache files, got "
                f"{len(val_set.topo_cache_files_used)}"
            )

    # Publish stats only after every requested split has been prepared
    # successfully, so a failed validation-cache pass cannot look complete.
    stats_value = args.stats_out or str(cfg.DATA.TOPO_STATS_PATH)
    if not str(stats_value).strip():
        parser.error("--stats-out or DATA.TOPO_STATS_PATH is required")
    stats_path = resolve_project_path(stats_value)
    stats = write_topology_stats(
        train_cache_files,
        Path(train_set.topo_cache_dir),
        stats_path,
        topo_dim=int(cfg.MODEL.TOPO_FEAT_DIM),
        cache_version=str(cfg.DATA.TOPO_CACHE_VERSION),
        eps=float(cfg.DATA.TOPO_NORM_EPS),
    )
    print(
        f"Saved train-only topology stats: {stats_path} "
        f"(samples={stats['sample_count']}, digest={stats['cache_digest_sha256']})"
    )

    print("Topology cache preparation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
