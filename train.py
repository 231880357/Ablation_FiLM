import argparse
import time
import os
import numpy as np
from tqdm import tqdm

from defaults import get_cfg_defaults
from dataset import Lung250MDataset, KittiDataset
from ppwc import Topo9_PointPWC, multiScaleLoss, topo_pyramid_loss


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
    device = cfg.DEVICE
    pin_memory = bool(cfg.DATA.PIN_MEMORY and device.startswith('cuda'))
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
    if getattr(args, 'dataset', 'lung') == 'kitti':
        train_set = KittiDataset(cfg, args, phase='train', split='train')
    else:
        train_set = Lung250MDataset(cfg, args, phase='train', split='train')
    if args.debug:
        if hasattr(train_set, 'case_list'):
            train_set.case_list = train_set.case_list[:8]
        else:
            train_set.file_list = train_set.file_list[:8]
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **dataloader_kwargs)
    
    if getattr(args, 'dataset', 'lung') == 'kitti':
        val_set = KittiDataset(cfg, args, phase='test', split='val')
    else:
        val_set = Lung250MDataset(cfg, args, phase='test', split='val')
    if args.debug:
        if hasattr(val_set, 'case_list'):
            val_set.case_list = val_set.case_list[:8]
        else:
            val_set.file_list = val_set.file_list[:8]
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **dataloader_kwargs)

    # logging
    validation_log = np.zeros([num_epochs, 3])

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

        train_loss = np.mean(loss_values)
        validation_log[ep - 1, 0] = train_loss
        lr_scheduler.step()

        # Validation
        model.eval()
        epe_3d = 0
        epe_initial = 0
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
            val_bar.set_postfix(curr_epe=f"{err_per_sample.mean().item() * val_loader.dataset.norm_factor:.4f}")

        epe_3d = epe_3d / len(val_loader.dataset) * val_loader.dataset.norm_factor
        epe_initial = epe_initial / len(val_loader.dataset) * val_loader.dataset.norm_factor
        validation_log[ep - 1, 1:] = [epe_initial, epe_3d]

        end_time = time.time()
        print('epoch', ep, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.6f' % train_loss,
              'initial error', epe_initial, 'EPEs', epe_3d)

        np.save(os.path.join(out_folder, "validation_history.npy"), validation_log)
        torch.save(model.state_dict(), model_path)
        if ep % cfg.SOLVER.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), model_path_ep.format(ep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--config', default='config_ppwc_sup.yaml',
                        help="config file of the model (yaml)")
    parser.add_argument("--debug", default=False, help="whether to use debug mode", type=bool)
    parser.add_argument("--gpu", default="0", help="gpu to train on")
    parser.add_argument('-CTr', '--cloudfolder_train', default='../cloudsTr/coordinates',
                        help="folder containing (/case_???_{1,2}.pth)")
    parser.add_argument('-CVal', '--cloudfolder_val', default='../cloudsTs/coordinates',
                        help="folder containing (/case_???_{1,2}.pth)")
    parser.add_argument('-STr','--supfolder_train', default='../corrfieldFlowPcdTr', 
                        help='folder containing ground truth (.pth)')
    parser.add_argument('-SVal','--supfolder_val', default='../corrfieldFlowPcdTs', 
                        help='folder containing ground truth (.pth)')
    parser.add_argument('--dataset', default='lung', choices=['lung', 'kitti'],
                        help="dataset to use")
    parser.add_argument('--kitti_root', default='../mmdetection3d/data/kitti/training/velodyne',
                        help="folder containing kitti bin files")
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override batch size from config')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='override num_workers from config')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    train(cfg, args)
