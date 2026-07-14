import os
import glob
import uuid
import numpy as np
import torch
import torch.utils.data
try:
    import open3d as o3d
except ImportError:
    o3d = None
import numpy as _np
from kitti_odometry_utils import (
    build_odometry_pairs,
    deterministic_sample,
    load_calibration_transform,
    load_pose_matrices,
    load_velodyne_points,
    parse_sequence_ids,
    relative_velodyne_transform,
    transform_points,
)
from topology_preprocessing import (
    TopologyNormalizer,
    load_topology_cache_file,
    topology_cache_record_id,
    validate_cache_version,
    validate_topology_vector,
)
try:
    from .topology import compute_topo_features
except Exception:
    try:
        from topology import compute_topo_features
    except Exception:
        compute_topo_features = None


def _safe_compute_topo_features(
    point_cloud,
    topo_dim,
    sample_name='sample',
    strict=False,
):
    if topo_dim <= 0:
        return np.zeros(1, dtype=np.float32)

    if compute_topo_features is None:
        if strict:
            raise RuntimeError(
                f'Topology extraction is unavailable for {sample_name}'
            )
        return np.zeros(topo_dim, dtype=np.float32)

    try:
        topo_feat = compute_topo_features(point_cloud)
        if np.asarray(topo_feat).shape != (topo_dim,):
            message = (
                f"WARNING: Unexpected topology dim for {sample_name}. "
                f"Expected ({topo_dim},), got {np.asarray(topo_feat).shape}"
            )
            if strict:
                raise ValueError(message.removeprefix('WARNING: '))
            print(message)
            return np.zeros(topo_dim, dtype=np.float32)
        topo_feat = validate_topology_vector(topo_feat, topo_dim)
        if strict and np.all(topo_feat == 0):
            raise ValueError(
                f'Topology extraction returned an all-zero vector for {sample_name}'
            )
        return topo_feat
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f'Topology extraction failed for {sample_name}: {exc}'
            ) from exc
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


def _normalize_kitti_sequence_ids(sequence_ids):
    normalized = []
    for sequence_id in sequence_ids or []:
        value = str(sequence_id).strip()
        if value.isdigit():
            value = f'{int(value):02d}'
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _find_kitti_sequence_files(kitti_root, sequence_id):
    candidate_dirs = [
        os.path.join(kitti_root, 'sequences', sequence_id, 'velodyne'),
        os.path.join(kitti_root, sequence_id, 'velodyne'),
        os.path.join(kitti_root, sequence_id),
    ]
    files = []
    visited_dirs = set()
    for candidate_dir in candidate_dirs:
        absolute_dir = os.path.abspath(candidate_dir)
        if absolute_dir in visited_dirs:
            continue
        visited_dirs.add(absolute_dir)
        files.extend(glob.glob(os.path.join(absolute_dir, '*.bin')))
    return sorted(set(files))


def _split_kitti_files(files, split):
    split_idx = int(len(files) * 0.8)
    if split == 'train':
        return files[:split_idx]
    if split == 'val':
        return files[split_idx:]
    raise NotImplementedError()


class _TopologyCacheMixin:
    def _init_topology_cache(self, cfg, split_name, prepare_mode=False):
        self.topo_cache_enabled = bool(self.use_topo and cfg.DATA.TOPO_CACHE_ENABLED)
        self.topo_cache_in_memory = bool(cfg.DATA.TOPO_CACHE_IN_MEMORY)
        self.topo_cache_mem = {}
        self.topo_cache_files_used = set()
        self.topo_cache_dir = None
        self.topo_cache_version = validate_cache_version(
            getattr(cfg.DATA, 'TOPO_CACHE_VERSION', 'v2')
        )

        if prepare_mode:
            self.topology_normalizer = TopologyNormalizer(
                mode='none',
                topo_dim=int(self.topo_dim),
                cache_version=self.topo_cache_version,
                eps=float(getattr(cfg.DATA, 'TOPO_NORM_EPS', 1e-6)),
            )
        else:
            self.topology_normalizer = TopologyNormalizer.from_config(
                cfg,
                self.topo_dim,
                verify_cache_digest=False,
            )
        self.strict_topology_computation = bool(
            prepare_mode or self.topology_normalizer.enabled
        )
        self.verify_topology_cache_entries = bool(
            self.topology_normalizer.enabled
            and split_name.startswith('train')
            and not prepare_mode
        )

        if not self.topo_cache_enabled:
            if self.verify_topology_cache_entries:
                raise ValueError(
                    "Train-only z-score normalization requires DATA.TOPO_CACHE_ENABLED=True"
                )
            return

        cache_root = cfg.DATA.TOPO_CACHE_DIR
        if not os.path.isabs(cache_root):
            cache_root = os.path.join(os.path.dirname(__file__), cache_root)
        self.topo_cache_dir = os.path.join(
            cache_root,
            self.__class__.__name__,
            self.topo_cache_version,
            split_name,
        )
        os.makedirs(self.topo_cache_dir, exist_ok=True)

    def _topo_cache_path(self, cache_key, point_cloud, topo_dim):
        if self.topo_cache_dir is None:
            return None
        safe_name = topology_cache_record_id(
            cache_key,
            point_cloud,
            topo_dim,
            self.topo_cache_version,
        )
        return os.path.join(self.topo_cache_dir, f'{safe_name}.npy')

    def _normalize_topology(self, topo_feat):
        return self.topology_normalizer.normalize(topo_feat)

    def _verify_topology_cache_entry(self, cache_path, topo_feat):
        if self.verify_topology_cache_entries:
            self.topology_normalizer.verify_cache_entry(
                cache_path,
                self.topo_cache_dir,
                topo_feat,
            )

    def _get_cached_topology(self, cache_key, point_cloud, topo_dim, sample_name):
        if topo_dim <= 0:
            return np.zeros(1, dtype=np.float32)

        if not self.topo_cache_enabled:
            return _safe_compute_topo_features(
                point_cloud,
                topo_dim,
                sample_name,
                strict=self.strict_topology_computation,
            )

        cache_path = self._topo_cache_path(cache_key, point_cloud, topo_dim)
        memory_key = cache_path if cache_path is not None else cache_key
        if self.topo_cache_in_memory and memory_key in self.topo_cache_mem:
            topo_feat = self.topo_cache_mem[memory_key].copy()
            if cache_path is not None:
                self.topo_cache_files_used.add(os.path.abspath(cache_path))
            self._verify_topology_cache_entry(cache_path, topo_feat)
            return topo_feat

        if cache_path is not None and os.path.exists(cache_path):
            try:
                topo_feat = load_topology_cache_file(cache_path, topo_dim)
                if self.strict_topology_computation and np.all(topo_feat == 0):
                    raise ValueError('all-zero fallback topology vector')
            except Exception as exc:
                print(f'WARNING: Rebuilding invalid topology cache {cache_path}: {exc}')
            else:
                self.topo_cache_files_used.add(os.path.abspath(cache_path))
                self._verify_topology_cache_entry(cache_path, topo_feat)
                if self.topo_cache_in_memory:
                    self.topo_cache_mem[memory_key] = topo_feat
                return topo_feat.copy()

        topo_feat = _safe_compute_topo_features(
            point_cloud,
            topo_dim,
            sample_name,
            strict=self.strict_topology_computation,
        )
        topo_feat = validate_topology_vector(topo_feat, topo_dim)
        if cache_path is not None:
            tmp_path = f'{cache_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npy'
            try:
                np.save(tmp_path, topo_feat)
                os.replace(tmp_path, cache_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            self.topo_cache_files_used.add(os.path.abspath(cache_path))
            self._verify_topology_cache_entry(cache_path, topo_feat)
        if self.topo_cache_in_memory:
            self.topo_cache_mem[memory_key] = topo_feat
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
        self.idx_16k = torch.load(
            getattr(args, 'lung_index_file', 'ind_16384_train.pth'),
            map_location='cpu',
        )

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
        self._init_topology_cache(
            cfg,
            f'{split}_lung',
            prepare_mode=bool(getattr(args, 'prepare_topology_cache', False)),
        )
        if self.is_train and self.topology_normalizer.enabled:
            augmentation_method = str(self.augm_setting.METHOD)
            if augmentation_method == 'rigid_one' and abs(
                float(self.augm_setting.MAX_SCALE_OFFSET)
            ) > self.topology_normalizer.eps:
                raise ValueError(
                    "Topology z-score training requires AUGMENTATIONS.MAX_SCALE_OFFSET=0 "
                    "because cached topology is computed before augmentation"
                )
            if augmentation_method == 'multiscale_local_global':
                raise ValueError(
                    "Topology z-score training cannot reuse pre-augmentation topology "
                    "with multiscale_local_global augmentation"
                )

    def preflight_topology_cache_manifest(self):
        """Repair invalid train entries, then validate the complete stats manifest."""
        if not self.verify_topology_cache_entries:
            return

        expected_count = len(self) * 2
        stats_count = int(self.topology_normalizer.stats.get('sample_count', 0))
        if stats_count != expected_count:
            raise ValueError(
                f'Topology stats contain {stats_count} samples, but the current '
                f'training split requires {expected_count}. Run '
                'prepare_topology_cache.py again.'
            )

        print(
            f'Preflighting {expected_count} training topology cache entries '
            'against the z-score stats manifest'
        )
        self.topo_cache_files_used.clear()
        was_training = self.is_train
        self.is_train = False
        try:
            for index in range(len(self)):
                self[index]
        finally:
            self.is_train = was_training

        cache_root = os.path.abspath(self.topo_cache_dir)
        active_entries = {
            os.path.relpath(path, cache_root).replace(os.sep, '/')
            for path in self.topo_cache_files_used
        }
        manifest_entries = set(self.topology_normalizer.entry_digests or {})
        if active_entries != manifest_entries:
            missing = sorted(active_entries - manifest_entries)
            stale = sorted(manifest_entries - active_entries)
            raise ValueError(
                'Current training cache identities do not match the topology stats '
                f'manifest (new={missing[:3]}, stale={stale[:3]}). Run '
                'prepare_topology_cache.py again.'
            )
        self.topology_normalizer.verify_cache_directory(self.topo_cache_dir)
        print('Training topology cache manifest and digest verified')

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
        topo_feat_src = self._normalize_topology(topo_feat_src)
        topo_feat_tgt = self._normalize_topology(topo_feat_tgt)

        if self.is_train:
            if self.augm_setting.METHOD == 'multiscale_local_global':
                if o3d is None:
                    raise ImportError(
                        'open3d is required for multiscale_local_global augmentation'
                    )
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
        self.sample_seed = int(getattr(args, 'kitti_seed', 0))
        self._init_topology_cache(
            cfg,
            f'{split}_kitti',
            prepare_mode=bool(getattr(args, 'prepare_topology_cache', False)),
        )

        if hasattr(args, 'kitti_root') and args.kitti_root:
            self.kitti_root = os.path.abspath(os.path.expanduser(args.kitti_root))
        else:
            self.kitti_root = os.path.abspath('../mmdetection3d/data/kitti/training/velodyne')

        self.sequence_ids = _normalize_kitti_sequence_ids(
            getattr(args, 'kitti_sequences', None)
        )
        if self.sequence_ids:
            self.file_list = []
            missing_sequences = []
            for sequence_id in self.sequence_ids:
                sequence_files = _find_kitti_sequence_files(self.kitti_root, sequence_id)
                if not sequence_files:
                    missing_sequences.append(sequence_id)
                    continue
                self.file_list.extend(_split_kitti_files(sequence_files, split))

            if missing_sequences:
                missing = ', '.join(missing_sequences)
                raise FileNotFoundError(
                    f'No KITTI odometry .bin files found for sequence(s) {missing} '
                    f'under {self.kitti_root}'
                )
            print(
                f"KITTI {split}: sequences={','.join(self.sequence_ids)}, "
                f'files={len(self.file_list)}'
            )
        else:
            all_files = sorted(glob.glob(os.path.join(self.kitti_root, '*.bin')))
            self.file_list = _split_kitti_files(all_files, split)

        if len(self.file_list) == 0:
            print(f"Warning: No bin files found in {self.kitti_root}")

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        pcd = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        relative_path = os.path.relpath(file_path, self.kitti_root)

        if pcd.shape[0] > 16384:
            pcd, _ = deterministic_sample(
                pcd,
                16384,
                f'kitti:{relative_path}',
                self.sample_seed,
            )
        else:
            pad = 16384 - pcd.shape[0]
            if pad > 0:
                pcd = np.pad(pcd, ((0, pad), (0, 0)), 'wrap')

        mean = np.mean(pcd, axis=0)
        pcd_src = (pcd - mean) / self.norm_factor
        pcd_tgt = pcd_src.copy()
        lm_src = pcd_src.copy()
        lm_tgt = pcd_tgt.copy()

        cache_key = f"kitti_{os.path.splitext(relative_path)[0]}"
        topo_feat_src = self._get_cached_topology(
            cache_key, pcd_src, self.topo_dim, os.path.basename(file_path)
        )
        topo_feat_tgt = topo_feat_src.copy()
        topo_feat_src = self._normalize_topology(topo_feat_src)
        topo_feat_tgt = self._normalize_topology(topo_feat_tgt)

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


class KittiOdometryDataset(_TopologyCacheMixin, torch.utils.data.Dataset):
    def __init__(self, cfg, args, phase, split):
        self.is_train = phase == 'train'
        self.split = split
        self.use_topo = cfg.MODEL.TOPO_FEAT_DIM > 0
        self.topo_dim = cfg.MODEL.TOPO_FEAT_DIM
        self.norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        self.num_points = int(getattr(args, 'odom_num_points', 8192))
        self.sample_seed = int(getattr(args, 'odom_seed', 0))
        self.odom_root = getattr(args, 'odom_root', 'D:/kitti_odometry')
        self.gap = int(getattr(args, 'odom_gap', 1))
        self._sequence_cache = {}
        self._init_topology_cache(
            cfg,
            f'{split}_kitti_odom',
            prepare_mode=bool(getattr(args, 'prepare_topology_cache', False)),
        )

        sequence_values = (
            getattr(args, 'odom_train_seqs', None)
            if split == 'train'
            else getattr(args, 'odom_val_seqs', None)
        )
        default_sequences = (
            ['00', '01', '02', '03', '04', '05', '06', '07']
            if split == 'train'
            else ['08', '09', '10']
        )
        self.sequence_ids = parse_sequence_ids(sequence_values, default_sequences)
        self.pair_list = build_odometry_pairs(
            self.odom_root,
            self.sequence_ids,
            gap=self.gap,
            max_pairs=getattr(args, 'odom_max_pairs', None),
            require_poses=True,
        )
        if not self.pair_list:
            raise RuntimeError(
                f'No KITTI odometry pairs found for split={split}, '
                f'sequences={self.sequence_ids}'
            )
        print(
            f"KITTI odometry {split}: sequences={','.join(self.sequence_ids)}, "
            f'pairs={len(self.pair_list)}, gap={self.gap}, points={self.num_points}'
        )

    def _sequence_geometry(self, pair):
        if pair.sequence not in self._sequence_cache:
            self._sequence_cache[pair.sequence] = (
                load_calibration_transform(pair.calib_path),
                load_pose_matrices(pair.pose_path),
            )
        return self._sequence_cache[pair.sequence]

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        src_full = load_velodyne_points(pair.src_bin)
        tgt_full = load_velodyne_points(pair.tgt_bin)
        src, _ = deterministic_sample(
            src_full,
            self.num_points,
            f'{pair.sequence}:{pair.src_idx}:src',
            self.sample_seed,
        )
        tgt, _ = deterministic_sample(
            tgt_full,
            self.num_points,
            f'{pair.sequence}:{pair.tgt_idx}:tgt',
            self.sample_seed,
        )

        calib_transform, poses = self._sequence_geometry(pair)
        relative_transform = relative_velodyne_transform(
            calib_transform, poses, pair.src_idx, pair.tgt_idx
        )
        aligned_src = transform_points(src, relative_transform)
        gt_flow = (aligned_src - src) / self.norm_factor

        target_mean = np.mean(tgt, axis=0)
        pcd_src = (src - target_mean) / self.norm_factor
        pcd_tgt = (tgt - target_mean) / self.norm_factor

        cache_prefix = (
            f'odom_{pair.sequence}_{pair.src_idx:06d}_to_{pair.tgt_idx:06d}'
            f'_gap{self.gap}_points{self.num_points}_seed{self.sample_seed}'
        )
        topo_feat_src = self._get_cached_topology(
            f'{cache_prefix}_src',
            pcd_src,
            self.topo_dim,
            f'odom-{pair.sequence}-{pair.src_idx:06d}-src',
        )
        topo_feat_tgt = self._get_cached_topology(
            f'{cache_prefix}_tgt',
            pcd_tgt,
            self.topo_dim,
            f'odom-{pair.sequence}-{pair.tgt_idx:06d}-tgt',
        )
        topo_feat_src = self._normalize_topology(topo_feat_src)
        topo_feat_tgt = self._normalize_topology(topo_feat_tgt)
        color_src, color_tgt, topo_feat_src, topo_feat_tgt = _compose_input_features(
            pcd_src, pcd_tgt, topo_feat_src, topo_feat_tgt, self.use_topo, self.topo_dim
        )

        outputs = (pcd_src, pcd_tgt, color_src, color_tgt, gt_flow)
        if not all(np.isfinite(value).all() for value in outputs):
            raise ValueError(
                f'Non-finite KITTI odometry sample: '
                f'{pair.sequence}:{pair.src_idx}->{pair.tgt_idx}'
            )
        return (
            np.float32(pcd_src),
            np.float32(pcd_tgt),
            np.float32(color_src),
            np.float32(color_tgt),
            np.float32(gt_flow),
            np.float32(topo_feat_src),
            np.float32(topo_feat_tgt),
            idx,
        )

    def __len__(self):
        return len(self.pair_list)
