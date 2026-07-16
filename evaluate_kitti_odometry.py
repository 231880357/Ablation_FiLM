import argparse
import csv
import json
from pathlib import Path

import numpy as np


THRESHOLDS = {
    'strict_absolute': 0.05,
    'strict_relative': 0.05,
    'relaxed_absolute': 0.10,
    'relaxed_relative': 0.10,
    'outlier_absolute': 0.30,
    'outlier_relative': 0.10,
}

SAMPLE_FIELDS = [
    'prediction_file',
    'sequence',
    'source_frame',
    'target_frame',
    'pose_available',
    'evaluated_points',
    'original_to_target_nn',
    'model_registered_to_target_nn',
    'nn_improvement',
    'nn_relative_improvement',
    'registration_improved',
    'original_chamfer',
    'chamfer_distance',
    'chamfer_improvement',
    'chamfer_improved',
    'epe3d',
    'acc3d_strict',
    'acc3d_relaxed',
    'outlier_rate',
]

SUMMARY_FIELDS = [
    'sequence',
    'sample_count',
    'chamfer_sample_count',
    'pose_sample_count',
    'point_evaluated_sample_count',
    'evaluated_point_count',
    'source_frame_min',
    'source_frame_max',
    'original_to_target_nn_mean',
    'model_registered_to_target_nn_mean',
    'nn_improvement_mean',
    'nn_relative_improvement_mean',
    'registration_improved_rate',
    'original_chamfer_mean',
    'chamfer_mean',
    'chamfer_min',
    'chamfer_25_percent',
    'chamfer_median',
    'chamfer_75_percent',
    'chamfer_max',
    'chamfer_improvement_mean',
    'chamfer_improved_rate',
    'epe3d_macro_mean',
    'epe3d_micro_mean',
    'acc3d_strict_macro_mean',
    'acc3d_strict_micro',
    'acc3d_relaxed_macro_mean',
    'acc3d_relaxed_micro',
    'outlier_rate_macro_mean',
    'outlier_rate_micro',
]


def _optional_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _mean(records, key):
    values = [record[key] for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _load_record(metrics_path):
    with metrics_path.open('r', encoding='utf-8') as stream:
        metrics = json.load(stream)

    required = ('sequence', 'source_frame', 'target_frame')
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f'missing metrics keys: {", ".join(missing)}')

    original_nn = _optional_float(metrics.get('original_to_target_nn'))
    model_nn = _optional_float(metrics.get('model_registered_to_target_nn'))
    nn_improvement = None
    nn_relative_improvement = None
    registration_improved = None
    if original_nn is not None and model_nn is not None:
        nn_improvement = original_nn - model_nn
        nn_relative_improvement = (
            nn_improvement / original_nn if original_nn > 0 else None
        )
        registration_improved = bool(model_nn < original_nn)

    original_chamfer = _optional_float(metrics.get('original_chamfer'))
    chamfer_distance = _optional_float(metrics.get('model_chamfer'))
    chamfer_improvement = None
    chamfer_improved = None
    if original_chamfer is not None and chamfer_distance is not None:
        chamfer_improvement = original_chamfer - chamfer_distance
        chamfer_improved = bool(chamfer_distance < original_chamfer)

    csv_path = Path(str(metrics_path)[:-len('.metrics.json')] + '.csv')

    record = {
        'prediction_file': csv_path.name,
        'sequence': f"{int(metrics['sequence']):02d}",
        'source_frame': int(metrics['source_frame']),
        'target_frame': int(metrics['target_frame']),
        'pose_available': bool(metrics.get('pose_available', False)),
        'evaluated_points': 0,
        'original_to_target_nn': original_nn,
        'model_registered_to_target_nn': model_nn,
        'nn_improvement': nn_improvement,
        'nn_relative_improvement': nn_relative_improvement,
        'registration_improved': registration_improved,
        'original_chamfer': original_chamfer,
        'chamfer_distance': chamfer_distance,
        'chamfer_improvement': chamfer_improvement,
        'chamfer_improved': chamfer_improved,
        'epe3d': None,
        'acc3d_strict': None,
        'acc3d_relaxed': None,
        'outlier_rate': None,
        '_error_sum': 0.0,
        '_strict_count': 0,
        '_relaxed_count': 0,
        '_outlier_count': 0,
    }

    if not record['pose_available']:
        return record, None

    if not csv_path.is_file():
        return record, f'missing CSV for pose evaluation: {csv_path}'

    values = np.atleast_2d(np.loadtxt(csv_path, delimiter=',', dtype=np.float64))
    if values.shape[1] < 9:
        return record, f'expected at least 9 CSV columns: {csv_path}'
    if not np.isfinite(values[:, :9]).all():
        return record, f'non-finite prediction values: {csv_path}'

    registered = values[:, :3]
    source = values[:, 3:6]
    pose_aligned = values[:, 6:9]
    errors = np.linalg.norm(registered - pose_aligned, axis=1)
    gt_flow_lengths = np.linalg.norm(pose_aligned - source, axis=1)
    relative_errors = errors / np.maximum(gt_flow_lengths, 1e-12)

    strict_mask = (
        (errors <= THRESHOLDS['strict_absolute'])
        | (relative_errors <= THRESHOLDS['strict_relative'])
    )
    relaxed_mask = (
        (errors <= THRESHOLDS['relaxed_absolute'])
        | (relative_errors <= THRESHOLDS['relaxed_relative'])
    )
    outlier_mask = (
        (errors > THRESHOLDS['outlier_absolute'])
        | (relative_errors > THRESHOLDS['outlier_relative'])
    )

    point_count = int(len(errors))
    record.update(
        {
            'evaluated_points': point_count,
            'epe3d': float(np.mean(errors)),
            'acc3d_strict': float(np.mean(strict_mask)),
            'acc3d_relaxed': float(np.mean(relaxed_mask)),
            'outlier_rate': float(np.mean(outlier_mask)),
            '_error_sum': float(np.sum(errors)),
            '_strict_count': int(np.count_nonzero(strict_mask)),
            '_relaxed_count': int(np.count_nonzero(relaxed_mask)),
            '_outlier_count': int(np.count_nonzero(outlier_mask)),
        }
    )
    return record, None


def _summarize(records, sequence='ALL'):
    point_records = [record for record in records if record['evaluated_points'] > 0]
    point_count = sum(record['evaluated_points'] for record in point_records)
    source_frames = [record['source_frame'] for record in records]
    chamfer_values = [
        record['chamfer_distance']
        for record in records
        if record['chamfer_distance'] is not None
    ]

    def chamfer_stat(function):
        return float(function(chamfer_values)) if chamfer_values else None

    def micro_rate(key):
        if point_count == 0:
            return None
        return float(sum(record[key] for record in point_records) / point_count)

    return {
        'sequence': sequence,
        'sample_count': len(records),
        'chamfer_sample_count': len(chamfer_values),
        'pose_sample_count': sum(record['pose_available'] for record in records),
        'point_evaluated_sample_count': len(point_records),
        'evaluated_point_count': point_count,
        'source_frame_min': min(source_frames) if source_frames else None,
        'source_frame_max': max(source_frames) if source_frames else None,
        'original_to_target_nn_mean': _mean(records, 'original_to_target_nn'),
        'model_registered_to_target_nn_mean': _mean(records, 'model_registered_to_target_nn'),
        'nn_improvement_mean': _mean(records, 'nn_improvement'),
        'nn_relative_improvement_mean': _mean(records, 'nn_relative_improvement'),
        'registration_improved_rate': _mean(records, 'registration_improved'),
        'original_chamfer_mean': _mean(records, 'original_chamfer'),
        'chamfer_mean': chamfer_stat(np.mean),
        'chamfer_min': chamfer_stat(np.min),
        'chamfer_25_percent': chamfer_stat(lambda values: np.quantile(values, 0.25)),
        'chamfer_median': chamfer_stat(np.median),
        'chamfer_75_percent': chamfer_stat(lambda values: np.quantile(values, 0.75)),
        'chamfer_max': chamfer_stat(np.max),
        'chamfer_improvement_mean': _mean(records, 'chamfer_improvement'),
        'chamfer_improved_rate': _mean(records, 'chamfer_improved'),
        'epe3d_macro_mean': _mean(point_records, 'epe3d'),
        'epe3d_micro_mean': micro_rate('_error_sum'),
        'acc3d_strict_macro_mean': _mean(point_records, 'acc3d_strict'),
        'acc3d_strict_micro': micro_rate('_strict_count'),
        'acc3d_relaxed_macro_mean': _mean(point_records, 'acc3d_relaxed'),
        'acc3d_relaxed_micro': micro_rate('_relaxed_count'),
        'outlier_rate_macro_mean': _mean(point_records, 'outlier_rate'),
        'outlier_rate_micro': micro_rate('_outlier_count'),
    }


def evaluate_predictions(predictions):
    predictions = Path(predictions).expanduser().resolve()
    if not predictions.is_dir():
        raise FileNotFoundError(f'Prediction directory does not exist: {predictions}')

    records = []
    warnings = []
    for metrics_path in sorted(predictions.glob('*.metrics.json')):
        try:
            record, warning = _load_record(metrics_path)
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f'{metrics_path.name}: {exc}')
            continue
        records.append(record)
        if warning:
            warnings.append(warning)

    if not records:
        raise RuntimeError(f'No usable .metrics.json files found in {predictions}')

    grouped = {}
    for record in records:
        grouped.setdefault(record['sequence'], []).append(record)
    per_sequence = [
        _summarize(grouped[sequence], sequence)
        for sequence in sorted(grouped)
    ]
    overall = _summarize(records)
    overall['sequence_count'] = len(grouped)
    overall['sequences'] = sorted(grouped)
    report = {
        'predictions': str(predictions),
        'thresholds': THRESHOLDS,
        'overall': overall,
        'per_sequence': per_sequence,
        'warnings': warnings,
    }
    public_records = [
        {key: record.get(key) for key in SAMPLE_FIELDS}
        for record in records
    ]
    return report, public_records


def _write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(report, sample_records, output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'evaluation_summary.json'
    sequence_csv_path = output_dir / 'evaluation_by_sequence.csv'
    sample_csv_path = output_dir / 'evaluation_per_sample.csv'

    with json_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    _write_csv(sequence_csv_path, report['per_sequence'], SUMMARY_FIELDS)
    _write_csv(sample_csv_path, sample_records, SAMPLE_FIELDS)
    return json_path, sequence_csv_path, sample_csv_path


def main(args):
    report, sample_records = evaluate_predictions(args.outfolder)
    write_evaluation(report, sample_records, args.outfolder)
    overall = report['overall']
    for record in sample_records:
        if record['chamfer_distance'] is not None:
            print(f"{record['prediction_file']}: CD = {record['chamfer_distance']:.4f}")

    if overall['chamfer_sample_count'] == 0:
        print(f'No KITTI csv files with Chamfer metrics found in {args.outfolder}')
        return

    print('======= SUMMARY RESULTS ======')
    print('KITTI Testing Chamfer Distance')
    print(
        f"mean: {overall['chamfer_mean']:.4f}, "
        f"min: {overall['chamfer_min']:.4f}, "
        f"25%: {overall['chamfer_25_percent']:.4f}, "
        f"50%: {overall['chamfer_median']:.4f}, "
        f"75%: {overall['chamfer_75_percent']:.4f}, "
        f"max: {overall['chamfer_max']:.4f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate KITTI Odometry inference outputs'
    )
    parser.add_argument(
        '-o',
        '--outfolder',
        default='prediction_kitti_odom',
        help='directory containing odom_*.csv and odom_*.metrics.json files',
    )
    main(parser.parse_args())
