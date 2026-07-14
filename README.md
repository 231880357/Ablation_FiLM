# Topo - 基础拓扑感知场景流估计

## 项目概述

Topo是**拓扑感知场景流估计**的基础实现版本，基于PointPWC-Net架构，引入持久同调（Persistent Homology）特征作为点云的几何拓扑描述符。

## 核心特点

### 1. 拓扑特征提取
- 使用 **ripser** 库计算点云的持久同调（0维和1维）
- 每个点云提取 **6维拓扑特征**：
  - H0平均寿命、H0最大寿命、H0熵
  - H1平均寿命、H1最大寿命、H1熵
- 基于流形距离（Manifold Distance）构建距离矩阵

### 2. 基础拓扑融合
- 将6维拓扑特征与3维坐标拼接作为输入（共9维）
- 拓扑特征仅在输入层使用，不参与后续层级处理
- 通过FPS降采样同步传播拓扑特征

### 3. 架构组成
```
输入层: PointConvSceneFlowPWC8192
├── Level 0: Conv1d(3+topo_dim, 32)
├── Level 1-4: PointConvD (金字塔降采样)
├── Cost Volume: PointConvFlow
├── Flow Estimator: SceneFlowEstimatorPointConv
└── Warping + Upsampling
```

## 关键文件

| 文件 | 功能 |
|------|------|
| `ppwc.py` | PointPWC网络主模型 |
| `pointconv_util.py` | PointConv工具函数和模块 |
| `topology.py` | 拓扑特征计算（持久同调）|
| `dataset.py` | 数据加载（含拓扑特征预处理）|
| `train.py` | 训练脚本 |
| `inference.py` | 推理脚本 |

## 训练配置

```yaml
MODEL:
  TOPO_FEAT_DIM: 6    # 拓扑特征维度
  PPWC_SFPC_BN: true  # 使用BatchNorm
```

## 与后续版本的对比

| 特性 | Topo | Topo1 | Topo2+ |
|------|------|-------|--------|
| 拓扑输入 | ✅ 简单拼接 | ✅ 门控机制 | ✅ FiLM调制 |
| 层级传递 | L0 only | L0-L3分层 | L0-L4分层 |
| 拓扑耦合 | 无 | 软门控 | 深度耦合 |
| 流估计器 | 标准 | 门控 | FiLM条件化 |

## 使用说明

```bash
# 训练
python train.py --config config_ppwc_sup.yaml --gpu 0

# 推理
python inference.py --config config_ppwc_sup.yaml --model model_best.pth
```

### KITTI Odometry 连续帧训练

`kitti_odom` 使用官方连续帧和 pose 生成监督场景流。下面的配置固定使用 sequence `00-07`
训练、`08-10` 验证、相邻帧间隔 1、每帧确定性采样 8192 点。每次启动还会在
`train_out_kitti_odom/topo9_kitti_odom/` 生成一份 `training_run_*.md` 参数记录。

```bash
python -X utf8 train.py \
  --dataset kitti_odom \
  --odom-root ../kitti_odometry \
  --odom-train-seqs 00,01,02,03,04,05,06,07 \
  --odom-val-seqs 08,09,10 \
  --odom-gap 1 \
  --odom-num-points 8192 \
  --odom-seed 0 \
  --config config_ppwc_kitti_odom.yaml \
  --gpu 1
```

### KITTI Odometry 推理

以下命令对验证 sequence `08-10` 各推理前 20 个相邻帧 pair。推理的 `gap`、点数、
采样 seed 和配置必须与训练保持一致。

```bash
python -X utf8 inference.py \
  --dataset kitti_odom \
  --odom-root ../kitti_odometry \
  --seqs 08 09 10 \
  --start 0 \
  --count 20 \
  --gap 1 \
  --odom-num-points 8192 \
  --odom-seed 0 \
  --config config_ppwc_kitti_odom.yaml \
  -M train_out_kitti_odom/topo9_kitti_odom/model.pth \
  -O prediction_kitti_odom \
  --gpu 1
```

每个 pair 会生成一个 CSV 和一个 `.metrics.json`。对有公开 pose 的 sequence `00-10`，
指标文件还包含 `model_to_pose_epe` 和 pose 对齐后的最近邻误差。更完整的数据目录说明和
小规模测试命令见 [`README_kitti.md`](README_kitti.md)。

## 实验数据

暂无实验数据记录。

## 拓扑缓存与 FiLM 诊断

旧版 `config_ppwc_sup.yaml` 保持未归一化行为，用于复现已有 checkpoint。新实验先用
`config_ppwc_sup_toponorm.yaml` 生成内容寻址的原始缓存和仅基于训练集的 z-score 统计：

```bash
python -X utf8 prepare_topology_cache.py \
  --config config_ppwc_sup_toponorm.yaml \
  --cloudfolder-train /path/to/lung/cloudsTr/coordinates \
  --cloudfolder-val /path/to/lung/cloudsTs/coordinates \
  --supfolder-train /path/to/lung/corrfieldFlowPcdTr \
  --supfolder-val /path/to/lung/corrfieldFlowPcdTs \
  --lung-index-file /path/to/lung/ind_16384_train.pth
```

训练时使用相同配置和统计文件：

```bash
python -X utf8 train.py \
  --dataset lung \
  --config config_ppwc_sup_toponorm.yaml \
  --cloudfolder_train /path/to/lung/cloudsTr/coordinates \
  --cloudfolder_val /path/to/lung/cloudsTs/coordinates \
  --supfolder_train /path/to/lung/corrfieldFlowPcdTr \
  --supfolder_val /path/to/lung/corrfieldFlowPcdTs \
  --lung-index-file /path/to/lung/ind_16384_train.pth \
  --gpu 0
```

训练结束后检查原始缓存，并使用训练统计重放归一化后的 FiLM 输入：

```bash
python -X utf8 diagnose_topology_film.py \
  --cache-dir topo_cache/Lung250MDataset/v2/train_lung \
  --stats-file topo_cache/Lung250MDataset/v2/train_stats.json \
  --checkpoint train_out/sup_16k_rigidAug1-2_toponorm_v2/model.pth
```

诊断脚本检查损坏文件、维度错误、NaN/Inf、负数、全零回退值、异常量级和分布离群点；原始缓存分布不会被归一化改变，只有 FiLM replay 输入使用训练集 z-score。发现缓存错误或接近 0 的 FiLM 分组时退出码为 `1`，统计或 checkpoint 无法读取时为 `2`。
