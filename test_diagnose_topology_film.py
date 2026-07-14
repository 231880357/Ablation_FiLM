import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from diagnose_topology_film import inspect_film_checkpoint, inspect_topology_cache, main
from topology_preprocessing import write_topology_stats


class TopologyCacheInspectionTests(unittest.TestCase):
    def test_healthy_cache_has_no_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(8):
                np.save(root / f"{index}.npy", np.arange(1, 7, dtype=np.float32) + index)

            report, matrix = inspect_topology_cache(root)

        self.assertEqual(report["file_count"], 8)
        self.assertEqual(report["valid_count"], 8)
        self.assertEqual(report["errors"], [])
        self.assertEqual(matrix.shape, (8, 6))

    def test_cache_content_errors_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "zero.npy", np.zeros(6, dtype=np.float32))
            np.save(root / "negative.npy", np.array([1, 2, 3, -1, 5, 6], dtype=np.float32))
            np.save(root / "nan.npy", np.array([1, 2, 3, np.nan, 5, 6], dtype=np.float32))
            np.save(root / "wrong_shape.npy", np.ones((1, 6), dtype=np.float32))

            report, _ = inspect_topology_cache(root)

        codes = {finding["code"] for finding in report["errors"]}
        self.assertTrue({"all_zero", "negative", "non_finite", "shape_mismatch"} <= codes)


class FilmCheckpointInspectionTests(unittest.TestCase):
    def test_reports_parameters_and_replayed_outputs(self):
        import torch

        state_dict = {
            "level4.film_gen.0.weight": torch.eye(2),
            "level4.film_gen.0.bias": torch.zeros(2),
            "level4.film_gen.1.weight": torch.ones(2),
            "level4.film_gen.1.bias": torch.zeros(2),
            "level4.gamma_gen.weight": torch.zeros((2, 2)),
            "level4.gamma_gen.bias": torch.zeros(2),
            "level4.beta_gen.weight": torch.full((2, 2), 0.1),
            "level4.beta_gen.bias": torch.zeros(2),
        }
        topology_vectors = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(state_dict, checkpoint)

            report = inspect_film_checkpoint(checkpoint, topology_vectors)

        parameters = {group["name"]: group["stats"] for group in report["parameter_groups"]}
        self.assertTrue(parameters["gamma_gen parameters"]["near_zero"])
        self.assertFalse(parameters["beta_gen parameters"]["near_zero"])
        self.assertEqual(len(report["output_groups"]), 1)
        self.assertTrue(report["output_groups"][0]["gamma"]["near_zero"])
        self.assertFalse(report["output_groups"][0]["beta"]["near_zero"])

    def test_main_returns_zero_for_healthy_cache_and_nonzero_film(self):
        import torch

        state_dict = {
            "level4.film_gen.0.weight": torch.ones((4, 6)),
            "level4.film_gen.0.bias": torch.arange(4, dtype=torch.float32),
            "level4.film_gen.1.weight": torch.ones(4),
            "level4.film_gen.1.bias": torch.zeros(4),
            "level4.gamma_gen.weight": torch.full((2, 4), 0.1),
            "level4.gamma_gen.bias": torch.full((2,), 0.1),
            "level4.beta_gen.weight": torch.full((2, 4), 0.2),
            "level4.beta_gen.bias": torch.full((2,), 0.2),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            for index in range(8):
                np.save(
                    cache_dir / f"{index}.npy",
                    np.arange(1, 7, dtype=np.float32) + index,
                )
            checkpoint = root / "model.pth"
            torch.save(state_dict, checkpoint)
            stats_path = root / "stats.json"
            write_topology_stats(
                sorted(cache_dir.glob("*.npy")),
                cache_dir,
                stats_path,
                topo_dim=6,
                cache_version="v2",
            )
            np.save(cache_dir / "orphan.npy", np.full(6, 1000.0, dtype=np.float32))

            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                exit_code = main(
                    [
                        "--cache-dir",
                        str(cache_dir),
                        "--stats-file",
                        str(stats_path),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )

            np.save(cache_dir / "0.npy", np.full(6, 99.0, dtype=np.float32))
            digest_output = io.StringIO()
            with redirect_stdout(digest_output), redirect_stderr(digest_output):
                digest_exit_code = main(
                    [
                        "--cache-dir",
                        str(cache_dir),
                        "--stats-file",
                        str(stats_path),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )

        self.assertEqual(exit_code, 0, output.getvalue())
        self.assertIn("Result: OK", output.getvalue())
        self.assertIn("FiLM replay normalization: zscore", output.getvalue())
        self.assertIn("orphan_cache_files", output.getvalue())
        self.assertEqual(digest_exit_code, 2, digest_output.getvalue())
        self.assertIn("digest", digest_output.getvalue())


if __name__ == "__main__":
    unittest.main()
