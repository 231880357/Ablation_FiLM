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

## 实验数据

暂无实验数据记录。
