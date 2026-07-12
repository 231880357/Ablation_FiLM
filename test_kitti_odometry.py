import argparse
from types import SimpleNamespace

import numpy as np
import torch

from dataset import KittiOdometryDataset
from kitti_odometry_utils import (
    deterministic_sample,
    load_calibration_transform,
    load_pose_matrices,
    load_velodyne_points,
    nearest_neighbor_mean,
    relative_velodyne_transform,
    transform_points,
)


def _smoke_config(topo_dim):
    return SimpleNamespace(
        MODEL=SimpleNamespace(TOPO_FEAT_DIM=topo_dim),
        INPUT=SimpleNamespace(SCALE_NORM_FACTOR=100),
        DATA=SimpleNamespace(
            TOPO_CACHE_ENABLED=False,
            TOPO_CACHE_IN_MEMORY=False,
            TOPO_CACHE_DIR='topo_cache',
        ),
    )


def main(args):
    dataset_args = SimpleNamespace(
        odom_root=args.odom_root,
        odom_train_seqs=args.sequence,
        odom_val_seqs=args.sequence,
        odom_gap=args.gap,
        odom_max_pairs=args.count,
        odom_num_points=args.num_points,
        odom_seed=args.seed,
    )
    dataset = KittiOdometryDataset(
        _smoke_config(args.topo_dim), dataset_args, phase='train', split='train'
    )
    sample = dataset[0]
    repeated_sample = dataset[0]
    pcd_src, pcd_tgt, color_src, color_tgt, gt_flow, topo_src, topo_tgt, _ = sample

    expected_feature_dim = 3 + args.topo_dim if args.topo_dim > 0 else 3
    assert pcd_src.shape == (args.num_points, 3)
    assert pcd_tgt.shape == (args.num_points, 3)
    assert color_src.shape == (args.num_points, expected_feature_dim)
    assert color_tgt.shape == (args.num_points, expected_feature_dim)
    assert gt_flow.shape == (args.num_points, 3)
    assert topo_src.shape == topo_tgt.shape
    assert all(np.isfinite(value).all() for value in sample[:7])
    assert float(np.linalg.norm(gt_flow, axis=1).mean()) > 0.0
    assert all(np.array_equal(sample[i], repeated_sample[i]) for i in range(7))

    pair = dataset.pair_list[0]
    src, _ = deterministic_sample(
        load_velodyne_points(pair.src_bin),
        args.num_points,
        f'{pair.sequence}:{pair.src_idx}:src',
        args.seed,
    )
    tgt, _ = deterministic_sample(
        load_velodyne_points(pair.tgt_bin),
        args.num_points,
        f'{pair.sequence}:{pair.tgt_idx}:tgt',
        args.seed,
    )
    transform = relative_velodyne_transform(
        load_calibration_transform(pair.calib_path),
        load_pose_matrices(pair.pose_path),
        pair.src_idx,
        pair.tgt_idx,
    )
    aligned_src = transform_points(src, transform)
    original_nn = nearest_neighbor_mean(src, tgt)
    aligned_nn = nearest_neighbor_mean(aligned_src, tgt)
    assert aligned_nn < original_nn

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    smoke_model = torch.nn.Sequential(
        torch.nn.Linear(expected_feature_dim, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 3),
    )
    optimizer = torch.optim.Adam(smoke_model.parameters(), lr=1e-3)
    features = batch[2].float()
    target_flow = batch[4].float()
    prediction = smoke_model(features)
    loss = torch.nn.functional.mse_loss(prediction, target_flow)
    optimizer.zero_grad()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in smoke_model.parameters()
    )
    optimizer.step()

    print(
        f'KITTI odometry smoke test passed: sequence={pair.sequence}, '
        f'pair={pair.src_idx:06d}->{pair.tgt_idx:06d}, points={args.num_points}'
    )
    print(
        f'shapes: src={pcd_src.shape}, target={pcd_tgt.shape}, '
        f'features={color_src.shape}, flow={gt_flow.shape}'
    )
    print(f'NN mean: original={original_nn:.6f}, pose_aligned={aligned_nn:.6f}')
    print(f'CPU backward smoke loss: {loss.item():.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smoke-test KITTI odometry data loading')
    parser.add_argument('--odom-root', default='D:/kitti_odometry')
    parser.add_argument('--sequence', default='00')
    parser.add_argument('--gap', type=int, default=1)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--num-points', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--topo-dim', type=int, default=0)
    parsed_args = parser.parse_args()
    for name in ('gap', 'count', 'num_points'):
        if getattr(parsed_args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be at least 1')
    if parsed_args.topo_dim < 0:
        parser.error('--topo-dim must be non-negative')
    main(parsed_args)
