import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate_kitti_odometry import evaluate_predictions, main, write_evaluation


class KittiOdometryEvaluationTest(unittest.TestCase):
    def test_evaluation_aggregates_nn_and_pose_metrics(self):
        workspace = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=workspace) as temp_dir:
            predictions = Path(temp_dir)
            self._write_prediction(
                predictions,
                sequence='08',
                source_frame=0,
                original_nn=0.5,
                model_nn=0.1,
                prediction_error=0.0,
            )
            self._write_prediction(
                predictions,
                sequence='09',
                source_frame=10,
                original_nn=0.4,
                model_nn=0.6,
                prediction_error=0.2,
            )

            report, samples = evaluate_predictions(predictions)
            overall = report['overall']

            self.assertEqual(overall['sample_count'], 2)
            self.assertEqual(overall['sequence_count'], 2)
            self.assertEqual(overall['evaluated_point_count'], 8)
            self.assertAlmostEqual(overall['registration_improved_rate'], 0.5)
            self.assertEqual(overall['chamfer_sample_count'], 2)
            self.assertAlmostEqual(overall['chamfer_mean'], 0.35)
            self.assertAlmostEqual(overall['chamfer_min'], 0.1)
            self.assertAlmostEqual(overall['chamfer_median'], 0.35)
            self.assertAlmostEqual(overall['chamfer_max'], 0.6)
            self.assertAlmostEqual(overall['chamfer_improved_rate'], 0.5)
            self.assertAlmostEqual(overall['epe3d_micro_mean'], 0.1)
            self.assertAlmostEqual(overall['acc3d_strict_micro'], 0.5)
            self.assertAlmostEqual(overall['acc3d_relaxed_micro'], 0.5)
            self.assertAlmostEqual(overall['outlier_rate_micro'], 0.5)

            paths = write_evaluation(report, samples, predictions)
            self.assertTrue(all(path.is_file() for path in paths))

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(SimpleNamespace(outfolder=str(predictions)))
            lines = output.getvalue().strip().splitlines()
            self.assertEqual(
                lines[0], 'odom_08_000000_to_000001.csv: CD = 0.1000'
            )
            self.assertEqual(
                lines[1], 'odom_09_000010_to_000011.csv: CD = 0.6000'
            )
            self.assertEqual(lines[-3], '======= SUMMARY RESULTS ======')
            self.assertEqual(lines[-2], 'KITTI Testing Chamfer Distance')
            self.assertEqual(
                lines[-1],
                'mean: 0.3500, min: 0.1000, 25%: 0.2250, '
                '50%: 0.3500, 75%: 0.4750, max: 0.6000',
            )

    @staticmethod
    def _write_prediction(
        predictions,
        sequence,
        source_frame,
        original_nn,
        model_nn,
        prediction_error,
    ):
        target_frame = source_frame + 1
        stem = f'odom_{sequence}_{source_frame:06d}_to_{target_frame:06d}'
        source = np.zeros((4, 3), dtype=np.float64)
        pose_aligned = source.copy()
        pose_aligned[:, 0] = 1.0
        registered = pose_aligned.copy()
        registered[:, 1] += prediction_error
        np.savetxt(
            predictions / f'{stem}.csv',
            np.concatenate([registered, source, pose_aligned], axis=1),
            delimiter=',',
        )
        metrics = {
            'sequence': sequence,
            'source_frame': source_frame,
            'target_frame': target_frame,
            'pose_available': True,
            'original_to_target_nn': original_nn,
            'model_registered_to_target_nn': model_nn,
            'original_chamfer': original_nn,
            'model_chamfer': model_nn,
            'model_to_pose_epe': prediction_error,
        }
        (predictions / f'{stem}.metrics.json').write_text(
            json.dumps(metrics), encoding='utf-8'
        )


if __name__ == '__main__':
    unittest.main()
