import os
import numpy as np
import torch
import torch.utils.data
import open3d as o3d
import numpy as _np
import hashlib
try:
    from .topology import compute_topo_features
except Exception:
    try:
        from topology import compute_topo_features
    except Exception:
        compute_topo_features = None


def _safe_compute_topo_features(point_cloud, topo_dim, sample_name='sample'):
    if topo_dim <= 0:
        return np.zeros(1, dtype=np.float32)

    if compute_topo_features is None:
        return np.zeros(topo_dim, dtype=np.float32)

    try:
        topo_feat = compute_topo_features(point_cloud)
        if topo_feat.shape[0] != topo_dim:
            print(
                f"WARNING: Unexpected topology dim for {sample_name}. "
                f"Expected {topo_dim}, got {topo_feat.shape}"
            )
            return np.zeros(topo_dim, dtype=np.float32)
        return topo_feat.astype(np.float32)
    except Exception as exc:
        print(f"Topology extraction failed for {sample_name}: {exc}")
        return np.zeros(topo_dim, dtype=np.float32)


def _compose_input_features(pcd_src, pcd_tgt, topo_feat_src, topo_feat_tgt, use_topo, topo_dim):
    if use_topo:
        feat_src_tile = _np.tile(topo_feat_src.reshape(1, -1), (pcd_src.shape[0], 1))
        feat_tgt_tile = _np.tile(topo_feat_tgt.reshape(1, -1), (pcd_tgt.shape[0], 1))
        color_src = _np.concatenate([pcd_src, feat_src_tile], axis=1)
        color_tgt = _np.concatenate([pcd_tgt, feat_tgt_tile], axis=1)
        return color_src, color_tgt, topo_feat_src, topo_feat_tgt

    dummy_topo = np.zeros(1, dtype=np.float32)
    return pcd_src, pcd_tgt, dummy_topo, dummy_topo


class _TopologyCacheMixin:
    def _init_topology_cache(self, cfg, split_name):
        self.topo_cache_enabled = bool(self.use_topo and cfg.DATA.TOPO_CACHE_ENABLED)
        self.topo_cache_in_memory = bool(cfg.DATA.TOPO_CACHE_IN_MEMORY)
        self.topo_cache_mem = {}
        self.topo_cache_dir = None

        if not self.topo_cache_enabled:
            return

        cache_root = cfg.DATA.TOPO_CACHE_DIR
        if not os.path.isabs(cache_root):
            cache_root = os.path.join(os.path.dirname(__file__), cache_root)
        self.topo_cache_dir = os.path.join(cache_root, self.__class__.__name__, split_name)
        os.makedirs(self.topo_cache_dir, exist_ok=True)

    def _topo_cache_path(self, cache_key):
        if self.topo_cache_dir is None:
            return None
        safe_name = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
        return os.path.join(self.topo_cache_dir, f'{safe_name}.npy')

    def _get_cached_topology(self, cache_key, point_cloud, topo_dim, sample_name):
        if topo_dim <= 0:
            return np.zeros(1, dtype=np.float32)

        if not self.topo_cache_enabled:
            return _safe_compute_topo_features(point_cloud, topo_dim, sample_name)

        if self.topo_cache_in_memory and cache_key in self.topo_cache_mem:
            return self.topo_cache_mem[cache_key].copy()

        cache_path = self._topo_cache_path(cache_key)
        if cache_path is not None and os.path.exists(cache_path):
            topo_feat = np.load(cache_path).astype(np.float32)
            if self.topo_cache_in_memory:
                self.topo_cache_mem[cache_key] = topo_feat
            return topo_feat.copy()

        topo_feat = _safe_compute_topo_features(point_cloud, topo_dim, sample_name)
        if cache_path is not None:
            tmp_path = f'{cache_path}.tmp.{os.getpid()}.npy'
            np.save(tmp_path, topo_feat)
            os.replace(tmp_path, cache_path)
        if self.topo_cache_in_memory:
            self.topo_cache_mem[cache_key] = topo_feat
        return topo_feat.copy()


class Lung250MDataset(_TopologyCacheMixin, torch.utils.data.Dataset):
    def __init__(self, cfg, args, phase, split):
        self.is_train = True if phase == 'train' else False
        self.split = split

        if self.split == 'train':
            self.pcd_template = os.path.join(args.cloudfolder_train, 'case_{:03d}_{}.pth')
            self.gt_template = os.path.join(args.supfolder_train, 'case_{:03d}.pth')
        else:
            self.pcd_template = os.path.join(args.cloudfolder_val, 'case_{:03d}_{}.pth')
            self.gt_template = os.path.join(args.supfolder_val, 'case_{:03d}.pth')
        self.idx_16k = torch.load('../ind_16384_train.pth', map_location='cpu')

        if split == 'train':
            val_cases = np.array([2, 8, 54, 55, 56, 94, 97])
            self.case_list = np.arange(104)
            self.case_list = self.case_list[~np.isin(self.case_list, val_cases)]
        elif split == 'val':
            self.case_list = np.array([2, 8, 94, 97])
        else:
            raise NotImplementedError()

        self.norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        self.augm_setting = cfg.AUGMENTATIONS
        
        # Check if topology coupling is enabled
        self.use_topo = cfg.MODEL.TOPO_FEAT_DIM > 0
        self.topo_dim = cfg.MODEL.TOPO_FEAT_DIM
        self._init_topology_cache(cfg, f'{split}_lung')

    def __getitem__(self, idx):
        # load input pcds
        case = self.case_list[idx]
        pcd_tgt = torch.load(self.pcd_template.format(case, 1))[2]
        pcd_src = torch.load(self.pcd_template.format(case, 2))[2]
        idx_16k_tgt = self.idx_16k['all_ind_fix'][case]
        idx_16k_src = self.idx_16k['all_ind_mov'][case]
        pcd_tgt = pcd_tgt[idx_16k_tgt].float().numpy()
        pcd_src = pcd_src[idx_16k_src].float().numpy()
        corrfield_flow = torch.load(self.gt_template.format(case))['cloud_gt_mov']
        corrfield_flow = corrfield_flow[idx_16k_src].float().numpy()
        lm_src = pcd_src.copy()
        lm_tgt = corrfield_flow + lm_src

        # prealignment
        mean_tgt = np.mean(pcd_tgt, axis=0)
        std_tgt = np.std(pcd_tgt, axis=0)
        mean_src = np.mean(pcd_src, axis=0)
        std_src = np.std(pcd_src, axis=0)
        pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
        lm_src = (lm_src - mean_src) * std_tgt / std_src + mean_tgt

        # mean center and scale
        mean = np.mean(pcd_tgt, axis=0)
        pcd_tgt = (pcd_tgt - mean) / self.norm_factor
        pcd_src = (pcd_src - mean) / self.norm_factor
        lm_tgt = (lm_tgt - mean) / self.norm_factor
        lm_src = (lm_src - mean) / self.norm_factor
        gt_flow = lm_tgt - lm_src

        topo_feat_src = self._get_cached_topology(
            f'lung_case_{case:03d}_src', pcd_src, self.topo_dim, f'lung-src-{case}'
        )
        topo_feat_tgt = self._get_cached_topology(
            f'lung_case_{case:03d}_tgt', pcd_tgt, self.topo_dim, f'lung-tgt-{case}'
        )

        if self.is_train:
            if self.augm_setting.METHOD == 'multiscale_local_global':
                if np.random.uniform() < 0.5:
                    pcd = pcd_src
                    feat_for_augm = topo_feat_src
                else:
                    pcd = pcd_tgt
                    feat_for_augm = topo_feat_tgt

                setting = self.augm_setting
                num_control_points_local = setting.NUM_CONTROL_POINTS_LOCAL
                max_control_shift_local = setting.MAX_CONTROL_SHIFT_LOCAL
                kernel_std_local = setting.KERNEL_STD_LOCAL
                global_grid_spacing = setting.GLOBAL_GRID_SPACING
                max_control_shift_global = setting.MAX_CONTROL_SHIFT_GLOBAL
                kernel_std_global = setting.KERNEL_STD_GLOBAL

                local_control_idx = np.random.permutation(pcd.shape[0])[:num_control_points_local]
                local_control_shifts = np.random.uniform(-1., 1., (num_control_points_local, 3)) * max_control_shift_local
                local_control_pts = pcd[local_control_idx]
                sq_dist = np.sum(np.square(pcd[:, None] - local_control_pts[None]), axis=2)
                weights = np.exp(-0.5 * sq_dist / kernel_std_local ** 2)
                local_pcd_shifts = np.sum(weights[:, :, None] * local_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
                local_pcd_shifts = np.nan_to_num(local_pcd_shifts)
                pcd_augm = pcd + local_pcd_shifts

                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(pcd_augm)
                o3d_cloud, _, _ = o3d_cloud.voxel_down_sample_and_trace(global_grid_spacing,
                                                                        min_bound=np.array([-10., -10., -10.]),
                                                                        max_bound=np.array([10., 10., 10.]))

                global_control_pts = np.float32(np.asarray(o3d_cloud.points))
                global_control_shifts = np.random.uniform(-1, 1., (
                global_control_pts.shape[0], 3)) * max_control_shift_global
                sq_dist = np.sum(np.square(pcd_augm[:, None] - global_control_pts[None]), axis=2)
                weights = np.exp(-0.5 * sq_dist / kernel_std_global ** 2)
                global_pcd_shifts = np.sum(weights[:, :, None] * global_control_shifts[None], axis=1) / np.sum(
                    weights[:, :, None], axis=1)

                pcd_augm = pcd_augm + global_pcd_shifts

                gt_flow = pcd - pcd_augm
                permutation = np.random.permutation(16384)
                pcd_src = pcd_augm[permutation[:8192]]
                gt_flow = gt_flow[permutation[:8192]]
                pcd_tgt = pcd[permutation[8192:]]
                
                # After augmentation, recompute topology or use pre-augmentation topo
                topo_feat_src = feat_for_augm
                topo_feat_tgt = feat_for_augm

            elif self.augm_setting.METHOD == 'rigid_one':
                setting = self.augm_setting
                max_transl = setting.MAX_TRANSLATION
                scale_offset = setting.MAX_SCALE_OFFSET
                rot_max = setting.MAX_ROTATION_ANGLE
                transl = np.random.uniform(-1., 1., (1, 3)) * max_transl
                scale = np.random.uniform(1 - scale_offset, 1 + scale_offset, (1, 3))
                rot_angles = np.deg2rad(np.random.uniform(-rot_max, rot_max, 3))

                theta = rot_angles[0]
                rot_mat_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
                theta = rot_angles[1]
                rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                theta = rot_angles[2]
                rot_mat_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                rot_mat = np.dot(np.dot(rot_mat_x, rot_mat_y), rot_mat_z)

                if np.random.uniform() < 0.5:
                    pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
                    lm_src = np.dot(lm_src, rot_mat) * scale + transl
                    gt_flow = lm_tgt - lm_src

                    permutation = np.random.permutation(16384)
                    pcd_src = pcd_src[permutation[:8192]]
                    gt_flow = gt_flow[permutation[:8192]]
                    pcd_tgt = pcd_tgt[permutation[8192:]]

                else:
                    pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
                    lm_src = np.dot(lm_src, rot_mat) * scale + transl
                    pcd_tgt = np.dot(pcd_tgt, rot_mat) * scale + transl
                    lm_tgt = np.dot(lm_tgt, rot_mat) * scale + transl
                    gt_flow = lm_tgt - lm_src

                    permutation = np.random.permutation(16384)
                    pcd_src = pcd_src[permutation[:8192]]
                    gt_flow = gt_flow[permutation[:8192]]
                    pcd_tgt = pcd_tgt[permutation[8192:]]

            else:
                pcd_src = pcd_src[:8192]
                pcd_tgt = pcd_tgt[:8192]
                gt_flow = gt_flow[:8192]

        else:
            pcd_src = pcd_src[:8192]
            pcd_tgt = pcd_tgt[:8192]
            gt_flow = gt_flow[:8192]

        color_src, color_tgt, topo_feat_src, topo_feat_tgt = _compose_input_features(
            pcd_src, pcd_tgt, topo_feat_src, topo_feat_tgt, self.use_topo, self.topo_dim
        )

        return (
            np.float32(pcd_src),
            np.float32(pcd_tgt),
            np.float32(color_src),
            np.float32(color_tgt),
            np.float32(gt_flow),
            np.float32(topo_feat_src),
            np.float32(topo_feat_tgt),
            idx
        )

    def __len__(self):
        return len(self.case_list)


class KittiDataset(_TopologyCacheMixin, torch.utils.data.Dataset):
    def __init__(self, cfg, args, phase, split):
        self.is_train = phase == 'train'
        self.split = split
        self.use_topo = cfg.MODEL.TOPO_FEAT_DIM > 0
        self.topo_dim = cfg.MODEL.TOPO_FEAT_DIM
        self.norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        self.augm_setting = cfg.AUGMENTATIONS
        self._init_topology_cache(cfg, f'{split}_kitti')

        if hasattr(args, 'kitti_root') and args.kitti_root:
            self.kitti_root = args.kitti_root
        else:
            self.kitti_root = '../mmdetection3d/data/kitti/training/velodyne'

        import glob
        all_files = sorted(glob.glob(os.path.join(self.kitti_root, '*.bin')))
        split_idx = int(len(all_files) * 0.8)
        if split == 'train':
            self.file_list = all_files[:split_idx]
        elif split == 'val':
            self.file_list = all_files[split_idx:]
        else:
            raise NotImplementedError()

        if len(self.file_list) == 0:
            print(f"Warning: No bin files found in {self.kitti_root}")

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        pcd = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        if pcd.shape[0] > 16384:
            permutation = np.random.permutation(pcd.shape[0])
            pcd = pcd[permutation[:16384]]
        else:
            pad = 16384 - pcd.shape[0]
            if pad > 0:
                pcd = np.pad(pcd, ((0, pad), (0, 0)), 'wrap')

        mean = np.mean(pcd, axis=0)
        pcd_src = (pcd - mean) / self.norm_factor
        pcd_tgt = pcd_src.copy()
        lm_src = pcd_src.copy()
        lm_tgt = pcd_tgt.copy()

        cache_key = f"kitti_{os.path.splitext(os.path.basename(file_path))[0]}"
        topo_feat_src = self._get_cached_topology(
            cache_key, pcd_src, self.topo_dim, os.path.basename(file_path)
        )
        topo_feat_tgt = topo_feat_src.copy()

        pcd_base = pcd_src.copy()
        if self.augm_setting.METHOD == 'multiscale_local_global':
            setting = self.augm_setting
            num_control_points_local = setting.NUM_CONTROL_POINTS_LOCAL
            max_control_shift_local = setting.MAX_CONTROL_SHIFT_LOCAL
            kernel_std_local = setting.KERNEL_STD_LOCAL
            global_grid_spacing = setting.GLOBAL_GRID_SPACING
            max_control_shift_global = setting.MAX_CONTROL_SHIFT_GLOBAL
            kernel_std_global = setting.KERNEL_STD_GLOBAL

            local_control_idx = np.random.permutation(pcd_base.shape[0])[:num_control_points_local]
            local_control_shifts = np.random.uniform(-1., 1., (num_control_points_local, 3)) * max_control_shift_local
            local_control_pts = pcd_base[local_control_idx]
            sq_dist = np.sum(np.square(pcd_base[:, None] - local_control_pts[None]), axis=2)
            weights = np.exp(-0.5 * sq_dist / kernel_std_local ** 2)
            local_pcd_shifts = np.sum(weights[:, :, None] * local_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
            local_pcd_shifts = np.nan_to_num(local_pcd_shifts)
            pcd_augm = pcd_base + local_pcd_shifts

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(pcd_augm)
            o3d_cloud, _, _ = o3d_cloud.voxel_down_sample_and_trace(
                global_grid_spacing,
                min_bound=np.array([-10., -10., -10.]),
                max_bound=np.array([10., 10., 10.])
            )

            global_control_pts = np.float32(np.asarray(o3d_cloud.points))
            global_control_shifts = np.random.uniform(-1, 1., (global_control_pts.shape[0], 3)) * max_control_shift_global
            sq_dist = np.sum(np.square(pcd_augm[:, None] - global_control_pts[None]), axis=2)
            weights = np.exp(-0.5 * sq_dist / kernel_std_global ** 2)
            global_pcd_shifts = np.sum(weights[:, :, None] * global_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
            pcd_augm = pcd_augm + global_pcd_shifts

            gt_flow = pcd_base - pcd_augm
            permutation = np.random.permutation(16384)
            pcd_src = pcd_augm[permutation[:8192]]
            gt_flow = gt_flow[permutation[:8192]]
            pcd_tgt = pcd_base[permutation[8192:]]

        elif self.augm_setting.METHOD == 'rigid_one':
            setting = self.augm_setting
            max_transl = setting.MAX_TRANSLATION
            scale_offset = setting.MAX_SCALE_OFFSET
            rot_max = setting.MAX_ROTATION_ANGLE
            transl = np.random.uniform(-1., 1., (1, 3)) * max_transl
            scale = np.random.uniform(1 - scale_offset, 1 + scale_offset, (1, 3))
            rot_angles = np.deg2rad(np.random.uniform(-rot_max, rot_max, 3))

            theta = rot_angles[0]
            rot_mat_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
            theta = rot_angles[1]
            rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            theta = rot_angles[2]
            rot_mat_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rot_mat = np.dot(np.dot(rot_mat_x, rot_mat_y), rot_mat_z)

            pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
            lm_src = np.dot(lm_src, rot_mat) * scale + transl
            gt_flow = lm_tgt - lm_src

            permutation = np.random.permutation(16384)
            pcd_src = pcd_src[permutation[:8192]]
            gt_flow = gt_flow[permutation[:8192]]
            pcd_tgt = pcd_tgt[permutation[8192:]]
        else:
            pcd_src = pcd_src[:8192]
            pcd_tgt = pcd_tgt[:8192]
            gt_flow = (lm_tgt - lm_src)[:8192]

        color_src, color_tgt, topo_feat_src, topo_feat_tgt = _compose_input_features(
            pcd_src, pcd_tgt, topo_feat_src, topo_feat_tgt, self.use_topo, self.topo_dim
        )

        return (
            np.float32(pcd_src),
            np.float32(pcd_tgt),
            np.float32(color_src),
            np.float32(color_tgt),
            np.float32(gt_flow),
            np.float32(topo_feat_src),
            np.float32(topo_feat_tgt),
            idx
        )

    def __len__(self):
        return len(self.file_list)
