import argparse
import json
import os
import numpy as np
import glob

from defaults import get_cfg_defaults
from kitti_odometry_utils import (
    build_odometry_pairs,
    deterministic_sample,
    load_calibration_transform,
    load_pose_matrices,
    load_velodyne_points,
    nearest_neighbor_mean,
    relative_velodyne_transform,
    transform_points,
    transform_summary,
)

compute_topo_features = None


def prepare_topology_inputs(pcd_src, pcd_tgt, topo_dim, device):
    color_src = pcd_src
    color_tgt = pcd_tgt
    topo_src = torch.zeros(1, topo_dim, device=device)
    topo_tgt = torch.zeros(1, topo_dim, device=device)

    if topo_dim <= 0:
        return color_src, color_tgt, topo_src[:, :1], topo_tgt[:, :1]

    if compute_topo_features is None:
        zeros_src = torch.zeros(1, pcd_src.shape[1], topo_dim, device=device)
        zeros_tgt = torch.zeros(1, pcd_tgt.shape[1], topo_dim, device=device)
        return torch.cat([pcd_src, zeros_src], dim=2), torch.cat([pcd_tgt, zeros_tgt], dim=2), topo_src, topo_tgt

    try:
        feat_src = compute_topo_features(pcd_src[0].cpu().numpy())
        feat_tgt = compute_topo_features(pcd_tgt[0].cpu().numpy())
        t_feat_src = torch.from_numpy(feat_src).float().to(device)
        t_feat_tgt = torch.from_numpy(feat_tgt).float().to(device)
    except Exception as exc:
        print(f"Topology extraction failed during inference: {exc}")
        t_feat_src = torch.zeros(topo_dim, device=device)
        t_feat_tgt = torch.zeros(topo_dim, device=device)

    topo_src = t_feat_src.unsqueeze(0)
    topo_tgt = t_feat_tgt.unsqueeze(0)
    t_feat_src_exp = topo_src.unsqueeze(1).repeat(1, pcd_src.shape[1], 1)
    t_feat_tgt_exp = topo_tgt.unsqueeze(1).repeat(1, pcd_tgt.shape[1], 1)
    color_src = torch.cat([pcd_src, t_feat_src_exp], dim=2)
    color_tgt = torch.cat([pcd_tgt, t_feat_tgt_exp], dim=2)
    return color_src, color_tgt, topo_src, topo_tgt


def run_kitti_odometry_inference(args, cfg, model, device, use_amp):
    pairs = build_odometry_pairs(
        args.odom_root,
        args.seqs,
        gap=args.gap,
        require_poses=False,
        start=args.start,
        count_per_sequence=args.count,
    )
    if not pairs:
        raise RuntimeError('No KITTI odometry inference pairs were found')

    pairs_per_sequence = {}
    for pair in pairs:
        pairs_per_sequence[pair.sequence] = pairs_per_sequence.get(pair.sequence, 0) + 1
    coverage = ', '.join(
        f'{sequence}={count}' for sequence, count in pairs_per_sequence.items()
    )
    print(f'KITTI odometry inference coverage: total={len(pairs)}, {coverage}')

    sequence_cache = {}
    norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
    for pair_index, pair in enumerate(pairs, start=1):
        src_full = load_velodyne_points(pair.src_bin)
        tgt_full = load_velodyne_points(pair.tgt_bin)
        src_np, _ = deterministic_sample(
            src_full,
            args.odom_num_points,
            f'{pair.sequence}:{pair.src_idx}:src',
            args.odom_seed,
        )
        tgt_np, _ = deterministic_sample(
            tgt_full,
            args.odom_num_points,
            f'{pair.sequence}:{pair.tgt_idx}:tgt',
            args.odom_seed,
        )

        target_mean = np.mean(tgt_np, axis=0)
        pcd_src = torch.from_numpy((src_np - target_mean) / norm_factor).float().unsqueeze(0).to(device)
        pcd_tgt = torch.from_numpy((tgt_np - target_mean) / norm_factor).float().unsqueeze(0).to(device)
        color_src, color_tgt, topo_src, topo_tgt = prepare_topology_inputs(
            pcd_src, pcd_tgt, cfg.MODEL.TOPO_FEAT_DIM, device
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                pred_flows, _, _, _, _ = model(
                    pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                )
                pred_flow = pred_flows[0].permute(0, 2, 1)[0].float().cpu().numpy()

        pred_flow *= norm_factor
        registered_src = src_np + pred_flow
        output_columns = [registered_src, src_np]
        original_to_target = nearest_neighbor_mean(src_np, tgt_full)
        target_to_original = nearest_neighbor_mean(tgt_full, src_np)
        registered_to_target = nearest_neighbor_mean(registered_src, tgt_full)
        target_to_registered = nearest_neighbor_mean(tgt_full, registered_src)
        metrics = {
            'sequence': pair.sequence,
            'source_frame': pair.src_idx,
            'target_frame': pair.tgt_idx,
            'source_points': int(len(src_np)),
            'target_points': int(len(tgt_np)),
            'target_full_points': int(len(tgt_full)),
            'pose_available': bool(pair.has_pose),
            'gap': args.gap,
            'sample_seed': args.odom_seed,
            'original_to_target_nn': original_to_target,
            'target_to_original_nn': target_to_original,
            'original_chamfer': (original_to_target + target_to_original) / 2.0,
            'model_registered_to_target_nn': registered_to_target,
            'target_to_model_registered_nn': target_to_registered,
            'model_chamfer': (registered_to_target + target_to_registered) / 2.0,
        }

        if pair.has_pose:
            if pair.sequence not in sequence_cache:
                sequence_cache[pair.sequence] = (
                    load_calibration_transform(pair.calib_path),
                    load_pose_matrices(pair.pose_path),
                )
            calib_transform, poses = sequence_cache[pair.sequence]
            relative_transform = relative_velodyne_transform(
                calib_transform, poses, pair.src_idx, pair.tgt_idx
            )
            pose_aligned_src = transform_points(src_np, relative_transform)
            output_columns.append(pose_aligned_src)
            translation, rotation_degrees = transform_summary(relative_transform)
            metrics.update(
                {
                    'relative_translation': translation,
                    'relative_rotation_degrees': rotation_degrees,
                    'pose_aligned_to_target_nn': nearest_neighbor_mean(pose_aligned_src, tgt_full),
                    'model_to_pose_epe': float(
                        np.linalg.norm(registered_src - pose_aligned_src, axis=1).mean()
                    ),
                }
            )

        stem = f'odom_{pair.sequence}_{pair.src_idx:06d}_to_{pair.tgt_idx:06d}'
        csv_path = os.path.join(args.outfile, f'{stem}.csv')
        metrics_path = os.path.join(args.outfile, f'{stem}.metrics.json')
        np.savetxt(csv_path, np.concatenate(output_columns, axis=1), delimiter=',', fmt='%.6f')
        with open(metrics_path, 'w', encoding='utf-8') as stream:
            json.dump(metrics, stream, indent=2, ensure_ascii=False)
        print(f'[{pair_index}/{len(pairs)}] Saved {csv_path}')


def main(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    # computational stuff
    use_amp = cfg.USE_AMP
    device = torch.device('cuda:0')

    # model: Topo9
    print("Using Topo9 (Original Topo + L4-only sparse residual topology coupling)")
    model = Topo9_PointPWC(cfg)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # data
    cases = [2, 8, 54, 55, 56, 94, 97, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
             120, 121, 122, 123]

    model.eval()

    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)

    if getattr(args, 'dataset', 'lung') == 'kitti_odom':
        run_kitti_odometry_inference(args, cfg, model, device, use_amp)
        return

    if getattr(args, 'dataset', 'lung') == 'kitti':
        kitti_root = getattr(args, 'kitti_root', '../mmdetection3d/data/kitti/testing/velodyne')
        all_files = sorted(glob.glob(os.path.join(kitti_root, '*.bin')))
        if len(all_files) == 0:
            raise FileNotFoundError(f"No bin files found in {kitti_root}")

        for i in range(len(all_files) - 1):
            file_src = all_files[i]
            file_tgt = all_files[i + 1]
            pcd_src_np = np.fromfile(file_src, dtype=np.float32).reshape(-1, 4)[:, :3]
            pcd_tgt_np = np.fromfile(file_tgt, dtype=np.float32).reshape(-1, 4)[:, :3]

            if pcd_src_np.shape[0] > 8192:
                pcd_src_np = pcd_src_np[np.random.permutation(pcd_src_np.shape[0])[:8192]]
            else:
                pad = 8192 - pcd_src_np.shape[0]
                if pad > 0:
                    pcd_src_np = np.pad(pcd_src_np, ((0, pad), (0, 0)), 'wrap')

            if pcd_tgt_np.shape[0] > 8192:
                pcd_tgt_np = pcd_tgt_np[np.random.permutation(pcd_tgt_np.shape[0])[:8192]]
            else:
                pad = 8192 - pcd_tgt_np.shape[0]
                if pad > 0:
                    pcd_tgt_np = np.pad(pcd_tgt_np, ((0, pad), (0, 0)), 'wrap')

            pcd_src = torch.from_numpy(pcd_src_np).float().unsqueeze(0).to(device)
            pcd_tgt = torch.from_numpy(pcd_tgt_np).float().unsqueeze(0).to(device)
            pcd_src_orig = pcd_src.clone()

            norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
            mean = torch.mean(pcd_tgt, axis=1, keepdim=True)
            pcd_tgt = (pcd_tgt - mean) / norm_factor
            pcd_src = (pcd_src - mean) / norm_factor

            color_src, color_tgt, topo_src, topo_tgt = prepare_topology_inputs(
                pcd_src, pcd_tgt, cfg.MODEL.TOPO_FEAT_DIM, device
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred_flows, _, _, _, _ = model(
                        pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                    )
                    pred_flow = pred_flows[0].permute(0, 2, 1)

            pred_flow = pred_flow * norm_factor
            tensor_to_save = torch.cat((pcd_src_orig + pred_flow, pcd_src_orig), dim=2)[0]
            out_filename = os.path.join(args.outfile, f'kitti_{i:04d}_to_{i + 1:04d}.csv')
            np.savetxt(out_filename, tensor_to_save.cpu().numpy(), delimiter=",", fmt='%.3f')
            print(f"Saved {out_filename}")

        return

    for i, case in enumerate(cases):
        pcd_tgt = torch.load(os.path.join(args.cloudfolder,'case_{:03d}_{}.pth'.format(case, 1)))[0].float()
        pcd_src = torch.load(os.path.join(args.cloudfolder,'case_{:03d}_{}.pth'.format(case, 2)))[0].float()
        pcd_tgt = pcd_tgt.unsqueeze(0).to(device)
        pcd_src = pcd_src.unsqueeze(0).to(device)

        # prealignment
        pcd_src_orig = pcd_src.clone()
        mean_tgt = torch.mean(pcd_tgt, dim=1)
        std_tgt = torch.std(pcd_tgt, dim=1)
        mean_src = torch.mean(pcd_src, dim=1)
        std_src = torch.std(pcd_src, dim=1)
        pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
        pre_align_flow = pcd_src - pcd_src_orig

        # mean center and scale
        norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        mean = torch.mean(pcd_tgt, axis=1)
        pcd_tgt = (pcd_tgt - mean) / norm_factor
        pcd_src = (pcd_src - mean) / norm_factor

        color_src, color_tgt, topo_src, topo_tgt = prepare_topology_inputs(
            pcd_src, pcd_tgt, cfg.MODEL.TOPO_FEAT_DIM, device
        )

        # inference
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                pred_flows, _, _, _, _ = model(
                    pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                )
                pred_flow = pred_flows[0].permute(0, 2, 1)

        pred_flow = pred_flow * cfg.INPUT.SCALE_NORM_FACTOR + pre_align_flow

        tensor_to_save = torch.cat((pcd_src_orig + pred_flow, pcd_src_orig), dim=2)[0]
        np.savetxt(os.path.join(args.outfile, 'case_{:03d}.csv'.format(case)), 
                   tensor_to_save.cpu().numpy(), delimiter=",", fmt='%.3f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference of Topology-Coupled PointPWC on Lung250M-4B')

    parser.add_argument('-M', '--model', default='ppwc_sup.pth', help="model file (pth)")
    parser.add_argument('-C', '--cloudfolder', default='cloudsTs',
                        help="folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-O', '--outfile', default='predictions_sup.pth',
                        help="output file for keypoint displacement predictions")
    parser.add_argument('--config', default='config_ppwc_sup.yaml',
                        help="config file of the model (yaml)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="physical GPU index to expose to this process")
    parser.add_argument('--dataset', default='lung', choices=['lung', 'kitti', 'kitti_odom'],
                        help="dataset to use")
    parser.add_argument('--kitti_root', default='../mmdetection3d/data/kitti/testing/velodyne',
                        help="folder containing kitti bin files")
    parser.add_argument('--odom-root', dest='odom_root', default='D:/kitti_odometry',
                        help='KITTI odometry root containing the official extracted folders')
    parser.add_argument('--seqs', nargs='+', default=['00'],
                        help='odometry sequence IDs, for example: --seqs 00 01')
    parser.add_argument('--start', type=int, default=0,
                        help='first source frame position in each sequence')
    parser.add_argument('--count', type=int, default=None,
                        help='optional pair limit per sequence; omitted means all available pairs')
    parser.add_argument('--gap', type=int, default=1,
                        help='frame gap between source and target')
    parser.add_argument('--odom-num-points', dest='odom_num_points', type=int, default=8192,
                        help='deterministically sampled points per frame')
    parser.add_argument('--odom-seed', dest='odom_seed', type=int, default=0,
                        help='base seed for deterministic odometry sampling')

    args = parser.parse_args()

    if args.gpu < 0:
        parser.error('--gpu must be a non-negative physical GPU index')
    for name in ('start',):
        if getattr(args, name) < 0:
            parser.error(f'--{name} must be non-negative')
    if args.count is not None and args.count < 1:
        parser.error('--count must be at least 1 when provided')
    for name in ('gap', 'odom_num_points'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be at least 1')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Import every Torch-dependent project module only after selecting the GPU.
    import torch
    from ppwc import Topo9_PointPWC
    try:
        from topology import compute_topo_features
    except ImportError:
        compute_topo_features = None

    if not torch.cuda.is_available():
        parser.error(
            f'physical GPU {args.gpu} is unavailable; '
            'check --gpu and CUDA_VISIBLE_DEVICES'
        )
    torch.cuda.set_device(0)
    print(
        f'GPU selection: physical GPU {args.gpu} -> logical cuda:0 '
        f'({torch.cuda.get_device_name(0)})'
    )

    print(args)
    main(args)
