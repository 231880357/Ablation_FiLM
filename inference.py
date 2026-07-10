import argparse
import os
import numpy as np
import glob

from defaults import get_cfg_defaults
from ppwc import Topo9_PointPWC
try:
    from topology import compute_topo_features
except ImportError:
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


def main(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    # computational stuff
    use_amp = cfg.USE_AMP
    device = 'cuda'

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
    parser.add_argument("--gpu", default="0", help="gpu to train on")
    parser.add_argument('--dataset', default='lung', choices=['lung', 'kitti'],
                        help="dataset to use")
    parser.add_argument('--kitti_root', default='../mmdetection3d/data/kitti/testing/velodyne',
                        help="folder containing kitti bin files")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch

    print(args)
    main(args)
