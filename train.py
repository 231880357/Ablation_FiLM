import argparse
import time
import os
import numpy as np
from tqdm import tqdm

from defaults import get_cfg_defaults


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}')


def _cuda_memory_stats(device):
    torch.cuda.synchronize(device)
    mib = 1024 ** 2
    return {
        'allocated': torch.cuda.memory_allocated(device) / mib,
        'reserved': torch.cuda.memory_reserved(device) / mib,
        'peak_allocated': torch.cuda.max_memory_allocated(device) / mib,
        'peak_reserved': torch.cuda.max_memory_reserved(device) / mib,
    }


def _print_cuda_memory(prefix, device):
    stats = _cuda_memory_stats(device)
    print(
        f"{prefix}: allocated={stats['allocated']:.1f} MiB, "
        f"reserved={stats['reserved']:.1f} MiB, "
        f"peak_allocated={stats['peak_allocated']:.1f} MiB, "
        f"peak_reserved={stats['peak_reserved']:.1f} MiB"
    )


def _limit_dataset(dataset, limit):
    if hasattr(dataset, 'case_list'):
        dataset.case_list = dataset.case_list[:limit]
    elif hasattr(dataset, 'pair_list'):
        dataset.pair_list = dataset.pair_list[:limit]
    else:
        dataset.file_list = dataset.file_list[:limit]


def train(cfg, args):
    root = cfg.BASE_DIRECTORY
    exp_name = cfg.EXPERIMENT_NAME
    out_folder = os.path.join(root, exp_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    model_path = os.path.join(out_folder, 'model.pth')
    model_path_ep = os.path.join(out_folder, 'model_ep={}.pth')

    # hyperparameters
    init_lr = cfg.SOLVER.LEARNING_RATE
    num_epochs = cfg.SOLVER.NUM_EPOCHS
    lr_steps = cfg.SOLVER.LR_MILESTONES
    lr_gamma = cfg.SOLVER.LR_LAMBDA
    batch_size = args.batch_size if args.batch_size is not None else cfg.SOLVER.BATCH_SIZE

    # computational stuff
    use_amp = cfg.USE_AMP
    num_workers = args.num_workers if args.num_workers is not None else (0 if args.debug else cfg.NUM_WORKERS)
    device = torch.device('cuda:0' if cfg.DEVICE.startswith('cuda') else cfg.DEVICE)
    pin_memory = bool(cfg.DATA.PIN_MEMORY and device.type == 'cuda')
    persistent_workers = bool(cfg.DATA.PERSISTENT_WORKERS and num_workers > 0)
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
    }
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = cfg.DATA.PREFETCH_FACTOR

    # model: Topo9 (Original Topo + L4-only topology coupling)
    print("Using Topo9 (Original Topo + L4-only sparse residual topology coupling)")
    model = Topo9_PointPWC(cfg)
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), init_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if cfg.SOLVER.SCHEDULER == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, lr_gamma)
    else:
        raise ValueError()

    # datasets
    dataset_name = getattr(args, 'dataset', 'lung')
    if dataset_name == 'kitti':
        train_set = KittiDataset(cfg, args, phase='train', split='train')
    elif dataset_name == 'kitti_odom':
        train_set = KittiOdometryDataset(cfg, args, phase='train', split='train')
    else:
        train_set = Lung250MDataset(cfg, args, phase='train', split='train')
    if args.debug:
        _limit_dataset(train_set, 8)
    if len(train_set) < batch_size:
        raise ValueError(
            f'Training dataset has {len(train_set)} samples, fewer than batch_size={batch_size}'
        )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **dataloader_kwargs)

    vram_test = bool(getattr(args, 'vram_test', False))
    if vram_test:
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('VRAM test requires a CUDA device')
        val_loader = None
        torch.cuda.reset_peak_memory_stats(device)
        print(
            f"VRAM test on {torch.cuda.get_device_name(device)}: "
            f'batch_size={batch_size}, steps={args.vram_test_steps}'
        )
        _print_cuda_memory('VRAM baseline', device)
    else:
        if dataset_name == 'kitti':
            val_set = KittiDataset(cfg, args, phase='test', split='val')
        elif dataset_name == 'kitti_odom':
            val_set = KittiOdometryDataset(cfg, args, phase='test', split='val')
        else:
            val_set = Lung250MDataset(cfg, args, phase='test', split='val')
        if args.debug:
            _limit_dataset(val_set, 8)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **dataloader_kwargs)

    # logging
    validation_log = np.zeros([num_epochs, 3])

    completed_train_steps = 0
    stop_after_epoch = False
    for ep in range(1, num_epochs + 1):
        print('Started epoch {}/{}'.format(ep, num_epochs))
        model.train()
        loss_values = []
        start_time = time.time()

        lambda_topo = 0.01  # Auxiliary topology consistency loss weight
        
        train_bar = tqdm(train_loader, desc=f"Epoch {ep}/{num_epochs} [train]", leave=False)
        for it, data in enumerate(train_bar, 1):
            pcd_src, pcd_tgt, color_src, color_tgt, gt_flow, topo_src, topo_tgt, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            color_src = color_src.to(device)
            color_tgt = color_tgt.to(device)
            gt_flow = gt_flow.to(device)
            topo_src = topo_src.to(device)
            topo_tgt = topo_tgt.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_flows, fps_pc1_idxs, _, _, _ = model(
                    pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                )
                loss_flow = multiScaleLoss(pred_flows, gt_flow, fps_pc1_idxs)
                loss_topo = topo_pyramid_loss(pcd_src, pcd_tgt, pred_flows[0], k=20)
                loss = loss_flow + lambda_topo * loss_topo

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at iteration {it}, skipping batch")
                continue
                
            loss_values.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            loss = loss * cfg.SOLVER.LOSS_FACTOR
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            completed_train_steps += 1
            if vram_test:
                _print_cuda_memory(f'VRAM step {completed_train_steps}', device)
                if completed_train_steps >= args.vram_test_steps:
                    print(
                        'VRAM test complete. No validation or checkpoint was run; '
                        'peak values above include forward, backward, and optimizer state.'
                    )
                    return

            if args.max_train_steps is not None and completed_train_steps >= args.max_train_steps:
                stop_after_epoch = True
                break

        if not loss_values:
            raise RuntimeError('No finite training loss was produced in this epoch')
        train_loss = np.mean(loss_values)
        validation_log[ep - 1, 0] = train_loss
        lr_scheduler.step()

        # Validation
        model.eval()
        epe_3d = 0
        epe_initial = 0
        val_samples = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {ep}/{num_epochs} [val]", leave=False)
        for it, data in enumerate(val_bar, 1):
            pcd_src, pcd_tgt, color_src, color_tgt, gt_flow, topo_src, topo_tgt, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            color_src = color_src.to(device)
            color_tgt = color_tgt.to(device)
            topo_src = topo_src.to(device)
            topo_tgt = topo_tgt.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred_flows, _, _, _, _ = model(
                        pcd_src, pcd_tgt, color_src, color_tgt, topo_src, topo_tgt
                    )
                    pred_flow = pred_flows[0].permute(0, 2, 1)

            gt_flow = gt_flow.to(device)
            err_per_sample = (pred_flow - gt_flow).square().sum(dim=2).sqrt().mean(dim=1)
            epe_3d += err_per_sample.sum().item()
            epe_initial += gt_flow.square().sum(dim=2).sqrt().mean(dim=1).sum().item()
            val_samples += len(err_per_sample)
            val_bar.set_postfix(curr_epe=f"{err_per_sample.mean().item() * val_loader.dataset.norm_factor:.4f}")

            if args.max_val_steps is not None and it >= args.max_val_steps:
                break

        if val_samples == 0:
            raise RuntimeError('Validation loader produced no samples')
        epe_3d = epe_3d / val_samples * val_loader.dataset.norm_factor
        epe_initial = epe_initial / val_samples * val_loader.dataset.norm_factor
        validation_log[ep - 1, 1:] = [epe_initial, epe_3d]

        end_time = time.time()
        print('epoch', ep, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.6f' % train_loss,
              'initial error', epe_initial, 'EPEs', epe_3d)

        np.save(os.path.join(out_folder, "validation_history.npy"), validation_log)
        torch.save(model.state_dict(), model_path)
        if ep % cfg.SOLVER.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), model_path_ep.format(ep))
        if stop_after_epoch:
            print(f'Stopped after requested {completed_train_steps} training step(s)')
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--config', default='config_ppwc_sup.yaml',
                        help="config file of the model (yaml)")
    parser.add_argument('--debug', nargs='?', const=True, default=False, type=_parse_bool,
                        help='limit datasets to 8 samples; optionally pass true/false')
    parser.add_argument("--gpu", type=int, default=0,
                        help="physical GPU index to expose to this process")
    parser.add_argument('-CTr', '--cloudfolder_train', default='../cloudsTr/coordinates',
                        help="folder containing (/case_???_{1,2}.pth)")
    parser.add_argument('-CVal', '--cloudfolder_val', default='../cloudsTs/coordinates',
                        help="folder containing (/case_???_{1,2}.pth)")
    parser.add_argument('-STr','--supfolder_train', default='../corrfieldFlowPcdTr', 
                        help='folder containing ground truth (.pth)')
    parser.add_argument('-SVal','--supfolder_val', default='../corrfieldFlowPcdTs', 
                        help='folder containing ground truth (.pth)')
    parser.add_argument('--dataset', default='lung', choices=['lung', 'kitti', 'kitti_odom'],
                        help="dataset to use")
    parser.add_argument('--kitti_root', default='../mmdetection3d/data/kitti/training/velodyne',
                        help="folder containing kitti bin files")
    parser.add_argument('--kitti_sequences', nargs='+', default=None,
                        help='KITTI odometry sequence IDs, for example: 00 01')
    parser.add_argument('--odom-root', dest='odom_root', default='D:/kitti_odometry',
                        help='KITTI odometry root containing the official extracted folders')
    parser.add_argument('--odom-train-seqs', dest='odom_train_seqs', default='00,01,02,03,04,05,06,07',
                        help='comma-separated pose-supervised training sequences')
    parser.add_argument('--odom-val-seqs', dest='odom_val_seqs', default='08,09,10',
                        help='comma-separated pose-supervised validation sequences')
    parser.add_argument('--odom-gap', dest='odom_gap', type=int, default=1,
                        help='frame gap between source and target')
    parser.add_argument('--odom-max-pairs', dest='odom_max_pairs', type=int, default=None,
                        help='limit pairs per split for small tests')
    parser.add_argument('--odom-num-points', dest='odom_num_points', type=int, default=8192,
                        help='deterministically sampled points per frame')
    parser.add_argument('--odom-seed', dest='odom_seed', type=int, default=0,
                        help='base seed for deterministic odometry sampling')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override batch size from config')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='override num_workers from config')
    parser.add_argument('--vram_test', action='store_true',
                        help='run a short KITTI odometry 00/01 training VRAM test')
    parser.add_argument('--vram_test_steps', type=int, default=10,
                        help='number of optimizer steps for --vram_test')
    parser.add_argument('--max-train-steps', dest='max_train_steps', type=int, default=None,
                        help='stop after this many optimizer steps, then run validation and save')
    parser.add_argument('--max-val-steps', dest='max_val_steps', type=int, default=None,
                        help='limit validation batches for a small training test')

    args = parser.parse_args()

    if args.vram_test:
        if args.dataset not in {'kitti', 'kitti_odom'}:
            parser.error('--vram_test requires --dataset kitti or kitti_odom')
        if args.vram_test_steps < 1:
            parser.error('--vram_test_steps must be at least 1')
        if args.dataset == 'kitti':
            args.kitti_sequences = ['00', '01']
        else:
            args.odom_train_seqs = '00,01'

    for name in ('odom_gap', 'odom_num_points'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be at least 1')
    for name in ('odom_max_pairs', 'max_train_steps', 'max_val_steps'):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f'--{name.replace("_", "-")} must be at least 1')

    if args.gpu < 0:
        parser.error('--gpu must be a non-negative physical GPU index')

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Import every Torch-dependent project module only after selecting the GPU.
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from dataset import KittiDataset, KittiOdometryDataset, Lung250MDataset
    from ppwc import Topo9_PointPWC, multiScaleLoss, topo_pyramid_loss

    if cfg.DEVICE.startswith('cuda'):
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

    train(cfg, args)
