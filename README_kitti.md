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

推理示例：

```bash
cd /root/autodl-tmp/registration_models_new/Ablation_FiLM
source /root/miniconda3/bin/activate lung_env_1
python inference.py --config config_ppwc_kitti.yaml --dataset kitti --kitti_root ../mmdetection3d/data/kitti/testing/velodyne -M train_out_kitti/topo9_kitti_rigid/model.pth -O prediction_kitti --gpu 0
```

评估可直接复用仓库根目录的 `evaluate_kitti.py`。