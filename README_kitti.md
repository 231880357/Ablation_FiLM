# Ablation_FiLM KITTI 适配

本目录现已支持从 `../mmdetection3d/data/kitti` 读取 KITTI 点云进行训练与推理。

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
