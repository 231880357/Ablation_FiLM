# Ablation_FiLM KITTI 适配

本目录支持旧版 KITTI 扁平点云输入，以及官方 KITTI Odometry 连续帧训练与推理。

## D 盘 KITTI Odometry

本机默认数据根目录为：

```text
D:\kitti_odometry
├── data_odometry_velodyne\dataset\sequences\00-21\velodyne\*.bin
├── data_odometry_calib\dataset\sequences\00-21\calib.txt
└── data_odometry_poses\dataset\poses\00-10.txt
```

`kitti_odom` 使用真实连续帧 `t -> t + gap`。默认训练 sequence 为 `00-07`，验证为 `08-10`；`11-21` 没有公开 pose，只用于推理。

首先运行不依赖 PointNet2 CUDA 扩展的数据测试：

```powershell
python -X utf8 test_kitti_odometry.py `
  --odom-root D:\kitti_odometry `
  --sequence 00 `
  --count 2
```

执行一个训练 step 和一个验证 batch，并保存测试 checkpoint：

```powershell
python -X utf8 train.py `
  --dataset kitti_odom `
  --odom-root D:\kitti_odometry `
  --odom-train-seqs 00 `
  --odom-val-seqs 08 `
  --odom-max-pairs 4 `
  --max-train-steps 1 `
  --max-val-steps 1 `
  --batch_size 1 `
  --num_workers 0 `
  --config config_ppwc_kitti_odom.yaml `
  --gpu 0
```

正式训练：

```powershell
python -X utf8 train.py `
  --dataset kitti_odom `
  --odom-root D:\kitti_odometry `
  --config config_ppwc_kitti_odom.yaml `
  --gpu 0
```

对 sequence 00 的前三个 pair 推理：

```powershell
python -X utf8 inference.py `
  --dataset kitti_odom `
  --odom-root D:\kitti_odometry `
  --seqs 00 `
  --start 0 `
  --count 3 `
  --gap 1 `
  --config config_ppwc_kitti_odom.yaml `
  --model train_out_kitti_odom\topo9_kitti_odom\model.pth `
  --outfile prediction_kitti_odom `
  --gpu 0
```

每个推理 pair 输出一个 CSV 和一个 `.metrics.json`。CSV 前 3 列是模型配准后的 source，接下来 3 列是原始 source；sequence `00-10` 还会追加 3 列 pose-aligned source。

## 旧版 KITTI 输入

训练示例：

```bash
cd /root/autodl-tmp/registration_models_new/Ablation_FiLM
source /root/miniconda3/bin/activate lung_env_1
python train.py --config config_ppwc_kitti.yaml --dataset kitti --kitti_root ../mmdetection3d/data/kitti/training/velodyne --gpu 0
```

更快的训练示例：

```bash
cd /root/autodl-tmp/registration_models_new/Ablation_FiLM
source /root/miniconda3/bin/activate lung_env_1
python train.py --config config_ppwc_kitti.yaml --dataset kitti --kitti_root ../mmdetection3d/data/kitti/training/velodyne --batch_size 6 --num_workers 8 --gpu 0
```

首次训练会在 `Ablation_FiLM/topo_cache/` 下生成拓扑缓存，后续 epoch 会明显更快。

## 使用 KITTI Odometry 00、01 测试显存

数据目录应包含如下结构：

```text
/path/to/kitti_odometry/dataset/
└── sequences/
    ├── 00/velodyne/*.bin
    └── 01/velodyne/*.bin
```

在远程主机运行：

```bash
python train.py \
  --config config_ppwc_kitti.yaml \
  --dataset kitti \
  --kitti_root /path/to/kitti_odometry/dataset \
  --vram_test \
  --vram_test_steps 10 \
  --batch_size 6 \
  --num_workers 4 \
  --gpu 0
```

`--vram_test` 会固定选择 sequence `00` 和 `01`，执行指定数量的完整优化步骤，逐步打印 PyTorch 的当前与峰值显存，然后直接退出；不会执行验证或保存 checkpoint。需要查看 CUDA 上下文等非 PyTorch 分配的总显存时，可在另一个终端同时运行：

```bash
watch -n 0.5 nvidia-smi
```

推理示例：

```bash
cd /root/autodl-tmp/registration_models_new/Ablation_FiLM
source /root/miniconda3/bin/activate lung_env_1
python inference.py --config config_ppwc_kitti.yaml --dataset kitti --kitti_root ../mmdetection3d/data/kitti/testing/velodyne -M train_out_kitti/topo9_kitti_rigid/model.pth -O prediction_kitti --gpu 0
```

评估可直接复用仓库根目录的 `evaluate_kitti.py`。
