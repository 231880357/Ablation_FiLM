import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_KITTI_ODOMETRY_ROOT = Path('D:/kitti_odometry')


@dataclass(frozen=True)
class KittiOdometryLayout:
    root: Path
    sequences_dir: Path
    calib_sequences_dir: Path
    poses_dir: Path


@dataclass(frozen=True)
class KittiOdometryPair:
    sequence: str
    src_idx: int
    tgt_idx: int
    src_bin: Path
    tgt_bin: Path
    calib_path: Path
    pose_path: Optional[Path]

    @property
    def has_pose(self):
        return self.pose_path is not None and self.pose_path.is_file()


def parse_sequence_ids(values, default=None):
    if values is None:
        values = default or []
    if isinstance(values, str):
        values = [values]

    sequence_ids = []
    for item in values:
        for part in str(item).split(','):
            value = part.strip()
            if not value:
                continue
            if not value.isdigit():
                raise ValueError(f'Invalid KITTI sequence ID: {value!r}')
            sequence_id = f'{int(value):02d}'
            if sequence_id not in sequence_ids:
                sequence_ids.append(sequence_id)
    return sequence_ids


def _first_existing_directory(candidates, description):
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    searched = '\n  '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Cannot find {description}. Searched:\n  {searched}')


def resolve_kitti_odometry_layout(odom_root=DEFAULT_KITTI_ODOMETRY_ROOT):
    root = Path(odom_root).expanduser().resolve()
    sequences_dir = _first_existing_directory(
        [
            root / 'data_odometry_velodyne' / 'dataset' / 'sequences',
            root / 'dataset' / 'sequences',
            root / 'sequences',
            root,
        ],
        'KITTI odometry Velodyne sequences directory',
    )
    calib_sequences_dir = _first_existing_directory(
        [
            root / 'data_odometry_calib' / 'dataset' / 'sequences',
            root / 'calib' / 'sequences',
            root / 'sequences',
            sequences_dir,
        ],
        'KITTI odometry calibration sequences directory',
    )

    pose_candidates = [
        root / 'data_odometry_poses' / 'dataset' / 'poses',
        root / 'dataset' / 'poses',
        root / 'poses',
    ]
    poses_dir = next(
        (candidate.resolve() for candidate in pose_candidates if candidate.is_dir()),
        pose_candidates[0].resolve(),
    )
    return KittiOdometryLayout(root, sequences_dir, calib_sequences_dir, poses_dir)


def build_odometry_pairs(
    odom_root,
    sequence_ids,
    gap=1,
    max_pairs=None,
    require_poses=False,
    start=0,
    count_per_sequence=None,
):
    if gap < 1:
        raise ValueError('gap must be at least 1')
    if start < 0:
        raise ValueError('start must be non-negative')
    if max_pairs is not None and max_pairs < 1:
        raise ValueError('max_pairs must be at least 1')
    if count_per_sequence is not None and count_per_sequence < 1:
        raise ValueError('count_per_sequence must be at least 1')

    layout = resolve_kitti_odometry_layout(odom_root)
    pairs = []
    for sequence_id in parse_sequence_ids(sequence_ids):
        velodyne_dir = layout.sequences_dir / sequence_id / 'velodyne'
        calib_path = layout.calib_sequences_dir / sequence_id / 'calib.txt'
        pose_path = layout.poses_dir / f'{sequence_id}.txt'

        if not velodyne_dir.is_dir():
            raise FileNotFoundError(f'Missing Velodyne directory: {velodyne_dir}')
        if not calib_path.is_file():
            raise FileNotFoundError(f'Missing calibration file: {calib_path}')
        if require_poses and not pose_path.is_file():
            raise FileNotFoundError(f'Missing pose file for supervised training: {pose_path}')

        frame_files = sorted(velodyne_dir.glob('*.bin'))
        sequence_pairs = []
        for position in range(start, max(0, len(frame_files) - gap)):
            src_bin = frame_files[position]
            tgt_bin = frame_files[position + gap]
            sequence_pairs.append(
                KittiOdometryPair(
                    sequence=sequence_id,
                    src_idx=int(src_bin.stem),
                    tgt_idx=int(tgt_bin.stem),
                    src_bin=src_bin,
                    tgt_bin=tgt_bin,
                    calib_path=calib_path,
                    pose_path=pose_path if pose_path.is_file() else None,
                )
            )
            if count_per_sequence is not None and len(sequence_pairs) >= count_per_sequence:
                break

        pairs.extend(sequence_pairs)
        if max_pairs is not None and len(pairs) >= max_pairs:
            return pairs[:max_pairs]
    return pairs


def load_velodyne_points(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(f'Invalid KITTI Velodyne file: {path}')
    points = raw.reshape(-1, 4)[:, :3]
    if not np.isfinite(points).all():
        raise ValueError(f'Non-finite coordinates in KITTI Velodyne file: {path}')
    return points


def deterministic_sample(points, count, key, base_seed=0):
    if count < 1:
        raise ValueError('sample count must be at least 1')
    if len(points) == 0:
        raise ValueError('cannot sample an empty point cloud')

    digest = hashlib.blake2b(
        f'{base_seed}:{key}'.encode('utf-8'), digest_size=8
    ).digest()
    seed = int.from_bytes(digest, byteorder='little', signed=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return np.ascontiguousarray(points[indices], dtype=np.float32), indices


def load_calibration_transform(path):
    values = None
    with open(path, 'r', encoding='utf-8') as stream:
        for line in stream:
            key, separator, raw_values = line.partition(':')
            if separator and key.strip() in {'Tr', 'Tr_velo_to_cam'}:
                values = np.fromstring(raw_values, sep=' ', dtype=np.float64)
                break
    if values is None or values.size != 12:
        raise ValueError(f'Cannot parse Tr from calibration file: {path}')
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :] = values.reshape(3, 4)
    return transform


def load_pose_matrices(path):
    values = np.loadtxt(path, dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] != 12:
        raise ValueError(f'Expected 12 pose values per row in {path}, got {values.shape}')
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
    poses[:, :3, :] = values.reshape(-1, 3, 4)
    return poses


def relative_velodyne_transform(calib_transform, poses, src_idx, tgt_idx):
    if src_idx >= len(poses) or tgt_idx >= len(poses):
        raise IndexError(
            f'Pose index out of range: src={src_idx}, tgt={tgt_idx}, poses={len(poses)}'
        )
    return (
        np.linalg.inv(calib_transform)
        @ np.linalg.inv(poses[tgt_idx])
        @ poses[src_idx]
        @ calib_transform
    )


def transform_points(points, transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.asarray(points @ rotation.T + translation, dtype=np.float32)


def nearest_neighbor_mean(source, target):
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(target).query(source, k=1, workers=-1)
    return float(np.mean(distances))


def transform_summary(transform):
    translation = float(np.linalg.norm(transform[:3, 3]))
    cosine = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation_degrees = float(math.degrees(math.acos(cosine)))
    return translation, rotation_degrees
