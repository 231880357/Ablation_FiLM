import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import dataset as dataset_module
from dataset import Lung250MDataset, _TopologyCacheMixin
from topology_preprocessing import (
    TopologyNormalizer,
    load_topology_cache_file,
    topology_cache_record_id,
    write_topology_stats,
)


def _config(cache_dir, mode="none", stats_path=""):
    return SimpleNamespace(
        DATA=SimpleNamespace(
            TOPO_CACHE_ENABLED=True,
            TOPO_CACHE_IN_MEMORY=False,
            TOPO_CACHE_DIR=str(cache_dir),
            TOPO_CACHE_VERSION="v2",
            TOPO_NORMALIZATION=mode,
            TOPO_STATS_PATH=str(stats_path),
            TOPO_NORM_EPS=1e-6,
        )
    )


class _FakeTopologyDataset(_TopologyCacheMixin):
    def __init__(self, cfg, split="val", prepare=False):
        self.use_topo = True
        self.topo_dim = 2
        self._init_topology_cache(
            cfg,
            f"{split}_fake",
            prepare_mode=prepare,
        )


class TopologyCacheIdentityTests(unittest.TestCase):
    def test_identity_changes_for_each_identity_component(self):
        points = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        baseline = topology_cache_record_id("case_001_src", points, 6, "v2")
        self.assertEqual(
            baseline,
            topology_cache_record_id("case_001_src", points.copy(), 6, "v2"),
        )

        changed = points.copy()
        changed[0, 0] += 1
        candidates = {
            topology_cache_record_id("case_001_tgt", points, 6, "v2"),
            topology_cache_record_id("case_001_src", changed, 6, "v2"),
            topology_cache_record_id("case_001_src", points, 5, "v2"),
            topology_cache_record_id("case_001_src", points, 6, "v3"),
        }
        self.assertEqual(len(candidates), 4)
        self.assertNotIn(baseline, candidates)

    def test_invalid_cache_is_rebuilt_atomically(self):
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(Path(directory) / "cache")
            first = _FakeTopologyDataset(cfg)
            with patch.object(
                dataset_module,
                "_safe_compute_topo_features",
                return_value=np.array([1.0, 2.0], dtype=np.float32),
            ):
                first._get_cached_topology("sample", points, 2, "sample")

            cache_path = Path(first._topo_cache_path("sample", points, 2))
            cache_path.write_bytes(b"truncated")
            calls = []

            def recompute(*_args, **_kwargs):
                calls.append(True)
                return np.array([3.0, 4.0], dtype=np.float32)

            second = _FakeTopologyDataset(cfg)
            with patch.object(
                dataset_module,
                "_safe_compute_topo_features",
                side_effect=recompute,
            ):
                rebuilt = second._get_cached_topology("sample", points, 2, "sample")

            np.testing.assert_array_equal(rebuilt, [3.0, 4.0])
            np.testing.assert_array_equal(load_topology_cache_file(cache_path, 2), rebuilt)
            self.assertEqual(len(calls), 1)
            self.assertEqual(list(cache_path.parent.glob("*.tmp.*")), [])

    def test_wrong_shape_nonfinite_and_nonnumeric_caches_are_rebuilt(self):
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        invalid_vectors = {
            "wrong_shape": np.ones(3, dtype=np.float32),
            "nonfinite": np.array([1.0, np.nan], dtype=np.float32),
            "string": np.array(["1", "2"]),
            "boolean": np.array([True, False]),
            "complex": np.array([1 + 2j, 3 + 4j]),
        }
        for label, invalid in invalid_vectors.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                dataset = _FakeTopologyDataset(_config(Path(directory) / "cache"))
                cache_path = Path(dataset._topo_cache_path("sample", points, 2))
                np.save(cache_path, invalid)
                with patch.object(
                    dataset_module,
                    "_safe_compute_topo_features",
                    return_value=np.array([3.0, 4.0], dtype=np.float32),
                ) as recompute:
                    rebuilt = dataset._get_cached_topology(
                        "sample", points, 2, "sample"
                    )
                recompute.assert_called_once()
                np.testing.assert_array_equal(rebuilt, [3.0, 4.0])
                np.testing.assert_array_equal(
                    load_topology_cache_file(cache_path, 2),
                    rebuilt,
                )

    def test_strict_preparation_rebuilds_zero_fallback_and_never_caches_failure(self):
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        with tempfile.TemporaryDirectory() as directory:
            dataset = _FakeTopologyDataset(
                _config(Path(directory) / "cache"),
                prepare=True,
            )
            cache_path = Path(dataset._topo_cache_path("sample", points, 2))
            np.save(cache_path, np.zeros(2, dtype=np.float32))
            with patch.object(
                dataset_module,
                "_safe_compute_topo_features",
                return_value=np.array([3.0, 4.0], dtype=np.float32),
            ):
                rebuilt = dataset._get_cached_topology(
                    "sample", points, 2, "sample"
                )
            np.testing.assert_array_equal(rebuilt, [3.0, 4.0])

            failed_path = Path(dataset._topo_cache_path("failure", points, 2))
            with patch.object(
                dataset_module,
                "compute_topo_features",
                None,
            ):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    dataset._get_cached_topology(
                        "failure", points, 2, "failure"
                    )
            self.assertFalse(failed_path.exists())


class TopologyStatisticsTests(unittest.TestCase):
    def test_train_only_stats_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "train_lung"
            cache_dir.mkdir()
            train_a = cache_dir / "a.npy"
            train_b = cache_dir / "b.npy"
            orphan = cache_dir / "old.npy"
            np.save(train_a, np.array([1.0, 10.0], dtype=np.float32))
            np.save(train_b, np.array([3.0, 14.0], dtype=np.float32))
            np.save(orphan, np.array([1000.0, 1000.0], dtype=np.float32))
            stats_path = root / "stats.json"

            stats = write_topology_stats(
                [train_a, train_b],
                cache_dir,
                stats_path,
                topo_dim=2,
                cache_version="v2",
            )

            np.testing.assert_allclose(stats["mean"], [2.0, 12.0])
            np.testing.assert_allclose(stats["std"], [1.0, 2.0])
            self.assertEqual(stats["sample_count"], 2)
            self.assertEqual({entry["file"] for entry in stats["cache_entries"]}, {"a.npy", "b.npy"})

            normalizer = TopologyNormalizer.from_stats_file(stats_path, 2, "v2")
            normalized = normalizer.normalize_matrix(
                np.array([[1.0, 10.0], [3.0, 14.0]], dtype=np.float32)
            )
            np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=1e-7)
            np.testing.assert_allclose(normalized.std(axis=0), [1.0, 1.0], atol=1e-7)
            np.testing.assert_array_equal(np.load(train_a), [1.0, 10.0])

            # Unreferenced files do not affect a manifest-based digest.
            normalizer.verify_source_cache()
            np.save(orphan, np.array([-999.0, -999.0], dtype=np.float32))
            normalizer.verify_source_cache()

            np.save(train_a, np.array([2.0, 10.0], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "digest"):
                normalizer.verify_source_cache()

    def test_none_mode_does_not_require_stats_but_zscore_does(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            none_normalizer = TopologyNormalizer.from_config(
                _config(Path(directory) / "cache", mode="none", stats_path=missing),
                2,
            )
            np.testing.assert_array_equal(
                none_normalizer.normalize(np.array([1.0, 2.0], dtype=np.float32)),
                [1.0, 2.0],
            )
            disabled_normalizer = TopologyNormalizer.from_config(
                _config(Path(directory) / "cache", mode="none", stats_path=missing),
                0,
            )
            np.testing.assert_array_equal(
                disabled_normalizer.normalize(np.zeros(1, dtype=np.float32)),
                [0.0],
            )
            with self.assertRaises(FileNotFoundError):
                TopologyNormalizer.from_config(
                    _config(Path(directory) / "cache", mode="zscore", stats_path=missing),
                    2,
                )

    def test_stats_dimension_version_and_small_std_fail_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "train"
            cache_dir.mkdir()
            first = cache_dir / "a.npy"
            second = cache_dir / "b.npy"
            np.save(first, np.array([1.0, 10.0], dtype=np.float32))
            np.save(second, np.array([3.0, 14.0], dtype=np.float32))
            stats_path = root / "stats.json"
            write_topology_stats([first, second], cache_dir, stats_path, 2, "v2")

            with self.assertRaisesRegex(ValueError, "does not match model dim"):
                TopologyNormalizer.from_stats_file(stats_path, 3, "v2")
            with self.assertRaisesRegex(ValueError, "does not match expected version"):
                TopologyNormalizer.from_stats_file(stats_path, 2, "v3")

            with stats_path.open("r", encoding="utf-8") as stream:
                stats = json.load(stream)
            stats["std"][0] = 1e-9
            with stats_path.open("w", encoding="utf-8") as stream:
                json.dump(stats, stream)
            with self.assertRaisesRegex(ValueError, "near-zero"):
                TopologyNormalizer.from_stats_file(stats_path, 2, "v2")

    def test_train_manifest_allows_rebuild_but_rejects_changed_identity(self):
        point_a = np.arange(12, dtype=np.float32).reshape(4, 3)
        point_b = point_a + 10
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            prepare_cfg = _config(cache_root, mode="none")
            prepared = _FakeTopologyDataset(prepare_cfg, split="train", prepare=True)

            def features(_points, _dim, sample_name, **_kwargs):
                if sample_name == "a":
                    return np.array([1.0, 2.0], dtype=np.float32)
                return np.array([3.0, 6.0], dtype=np.float32)

            with patch.object(
                dataset_module,
                "_safe_compute_topo_features",
                side_effect=features,
            ):
                prepared._get_cached_topology("a", point_a, 2, "a")
                prepared._get_cached_topology("b", point_b, 2, "b")

            stats_path = root / "stats.json"
            write_topology_stats(
                [Path(path) for path in prepared.topo_cache_files_used],
                Path(prepared.topo_cache_dir),
                stats_path,
                topo_dim=2,
                cache_version="v2",
            )
            train_cfg = _config(cache_root, mode="zscore", stats_path=stats_path)
            training = _FakeTopologyDataset(train_cfg, split="train")

            expected_path = Path(training._topo_cache_path("a", point_a, 2))
            expected_path.write_bytes(b"broken")
            with patch.object(
                dataset_module,
                "_safe_compute_topo_features",
                return_value=np.array([1.0, 2.0], dtype=np.float32),
            ):
                rebuilt = training._get_cached_topology("a", point_a, 2, "a")
            np.testing.assert_array_equal(rebuilt, [1.0, 2.0])

            changed = point_a.copy()
            changed[0, 0] += 0.5
            with patch.object(
                dataset_module,
                "_safe_compute_topo_features",
                return_value=np.array([1.0, 2.0], dtype=np.float32),
            ):
                with self.assertRaisesRegex(ValueError, "not represented"):
                    training._get_cached_topology("a", changed, 2, "a")


class TopologyIntegrationTests(unittest.TestCase):
    def test_lung_training_preflight_checks_complete_manifest_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "train_lung"
            cache_dir.mkdir()
            src = cache_dir / "src.npy"
            tgt = cache_dir / "tgt.npy"
            np.save(src, np.array([1.0, 10.0], dtype=np.float32))
            np.save(tgt, np.array([3.0, 14.0], dtype=np.float32))
            stats_path = root / "stats.json"
            write_topology_stats([src, tgt], cache_dir, stats_path, 2, "v2")

            dataset = Lung250MDataset.__new__(Lung250MDataset)
            dataset.verify_topology_cache_entries = True
            dataset.topology_normalizer = TopologyNormalizer.from_stats_file(
                stats_path, 2, "v2"
            )
            dataset.topo_cache_dir = str(cache_dir)
            dataset.topo_cache_files_used = set()
            dataset.is_train = True
            dataset.case_list = np.array([0])

            def fake_getitem(instance, _index):
                instance.topo_cache_files_used.update(
                    {str(src.resolve()), str(tgt.resolve())}
                )
                return None

            with patch.object(Lung250MDataset, "__getitem__", fake_getitem):
                dataset.preflight_topology_cache_manifest()
            self.assertTrue(dataset.is_train)

    def test_dataset_and_inference_use_identical_normalization(self):
        import torch
        import inference

        raw = np.array([1.0, 14.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "train"
            cache_dir.mkdir()
            first = cache_dir / "a.npy"
            second = cache_dir / "b.npy"
            np.save(first, np.array([1.0, 10.0], dtype=np.float32))
            np.save(second, np.array([3.0, 14.0], dtype=np.float32))
            stats_path = root / "stats.json"
            write_topology_stats(
                [first, second], cache_dir, stats_path, 2, "v2"
            )
            normalizer = TopologyNormalizer.from_stats_file(stats_path, 2, "v2")

            fake_dataset = _FakeTopologyDataset(
                _config(root / "cache", mode="none")
            )
            fake_dataset.topology_normalizer = normalizer
            expected = fake_dataset._normalize_topology(raw)

            inference.torch = torch
            with patch.object(inference, "compute_topo_features", return_value=raw):
                points = torch.zeros((1, 3, 3), dtype=torch.float32)
                color_src, _, topo_src, _ = inference.prepare_topology_inputs(
                    points,
                    points,
                    topo_dim=2,
                    device=torch.device("cpu"),
                    topology_normalizer=normalizer,
                )

            np.testing.assert_allclose(topo_src.numpy()[0], expected)
            np.testing.assert_allclose(
                color_src.numpy()[0, :, -2:],
                np.tile(expected, (3, 1)),
            )
            with patch.object(
                inference,
                "compute_topo_features",
                return_value=np.zeros(2, dtype=np.float32),
            ):
                with self.assertRaisesRegex(RuntimeError, "normalization failed"):
                    inference.prepare_topology_inputs(
                        points,
                        points,
                        topo_dim=2,
                        device=torch.device("cpu"),
                        topology_normalizer=normalizer,
                    )

    def test_inference_none_mode_preserves_disabled_topology_behavior(self):
        import torch
        import inference

        inference.torch = torch
        points = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        normalizer = TopologyNormalizer(
            mode="none",
            topo_dim=0,
            cache_version="v2",
            eps=1e-6,
        )
        color_src, color_tgt, topo_src, topo_tgt = inference.prepare_topology_inputs(
            points,
            points,
            topo_dim=0,
            device=torch.device("cpu"),
            topology_normalizer=normalizer,
        )
        self.assertTrue(torch.equal(color_src, points))
        self.assertTrue(torch.equal(color_tgt, points))
        self.assertEqual(tuple(topo_src.shape), (1, 0))
        self.assertEqual(tuple(topo_tgt.shape), (1, 0))


if __name__ == "__main__":
    unittest.main()
