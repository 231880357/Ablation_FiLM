# KITTI Odometry 训练与预测迁移计划

## 1. 目标

将当前 PointPWC/FiLM 代码从原先的 KITTI detection 扁平文件夹输入，迁移到真正有时序关系的 KITTI Odometry 数据集上。

目标分三层：

- 使用现有模型在 odometry 连续帧上生成预测结果。
- 使用 `poses/00-10.txt` 生成 ego-motion baseline，并做指标评估与可视化。
- 新增 odometry 训练数据入口，支持后续 fine-tune 或重新训练。

当前数据根目录：

```text
D:\kitti_odometry
```

当前项目目录：

```text
D:\Ablation_FiLM
```

当前 odometry 数据没有 color/gray 图像，因此第一版只做 LiDAR 点云训练、预测、评估和可视化，不做 image overlay。

## 2. 数据集结构建议

保留 KITTI 官方解压结构，不复制 80GB 点云数据：

```text
D:\kitti_odometry
  data_odometry_velodyne
    dataset\sequences\00-21\velodyne\*.bin
  data_odometry_calib
    dataset\sequences\00-21\calib.txt
    dataset\sequences\00-21\times.txt
  data_odometry_poses
    dataset\poses\00.txt - 10.txt
  devkit_odometry
```

新增轻量 prepared 层，只保存 manifest：

```text
D:\kitti_odometry\prepared
  manifests
    train_pairs.csv
    val_pairs.csv
    infer_pairs.csv
```

manifest 字段：

```text
seq,src_idx,tgt_idx,src_bin,tgt_bin,calib,times,pose_file,has_pose
```

默认划分：

- 训练：sequence `00-07`
- 验证：sequence `08-10`
- 推理展示：sequence `00-21`
- `11-21` 没有公开 pose，只做预测展示，不做有监督训练和 pose 指标评估

默认 pair 规则：

```text
source = frame t
target = frame t + gap
gap = 1
```

后续可以用 `gap=2/5` 测试更大帧间运动。

## 3. 公共 Odometry 数据工具

建议新增一个公共工具模块，例如：

```text
kitti_odometry_utils.py
```

核心职责：

- 解析 odometry 根目录。
- 读取 `.bin` 点云，取 `x, y, z`。
- 读取 `calib.txt` 中的 `Tr`。
- 读取 `poses/*.txt`，每行转成 `4x4` 位姿矩阵。
- 根据 source/target pose 计算 LiDAR 坐标系下的相对变换。
- 提供确定性采样函数，保证同一个 pair 每次采样一致。

核心变换：

```python
T_src_to_tgt_velo = inv(Tr) @ inv(Pose_tgt) @ Pose_src @ Tr
pose_aligned_source = T_src_to_tgt_velo @ source
gt_flow = pose_aligned_source - source
```

说明：

- `Pose_src` 和 `Pose_tgt` 是 KITTI odometry pose 文件中的相机坐标系全局位姿。
- `Tr` 是 `calib.txt` 中 LiDAR 到相机的外参。
- `gt_flow` 是 ego-motion flow，主要适用于静态背景；动态车辆、行人会带来监督噪声。

## 4. 训练改动

在 `dataset.py` 中新增：

```python
class KittiOdometryDataset(torch.utils.data.Dataset)
```

返回格式保持和当前训练代码兼容：

```python
pcd_src, pcd_tgt, color_src, color_tgt, gt_flow, idx
```

数据处理规则：

- source 从 `src_bin` 采样 8192 点。
- target 从 `tgt_bin` 采样 8192 点。
- source 和 target 使用同一个 target mean 归一化。
- 点云坐标除以 `cfg.INPUT.SCALE_NORM_FACTOR`。
- `gt_flow` 使用 pose 生成，并同样除以 `cfg.INPUT.SCALE_NORM_FACTOR`。
- `color_src = pcd_src`，`color_tgt = pcd_tgt`，保持现有模型接口不变。

需要在 `train.py` 中新增参数：

```text
--dataset kitti_odom
--odom-root D:\kitti_odometry
--odom-train-seqs 00,01,02,03,04,05,06,07
--odom-val-seqs 08,09,10
--odom-gap 1
--odom-max-pairs
```

训练入口逻辑：

- `dataset == lung`：保持现有 Lung 逻辑。
- `dataset == kitti`：保持当前旧 KITTI 合成/增强逻辑。
- `dataset == kitti_odom`：使用 `KittiOdometryDataset`。

建议新增配置文件：

```text
config_ppwc_kitti_odom.yaml
```

第一版可以继承现有 supervised 配置，重点确认：

- `INPUT.SCALE_NORM_FACTOR: 100`
- `SOLVER.BATCH_SIZE` 根据显存调整，默认 4
- `AUGMENTATIONS.METHOD` 对 odometry pose-supervised 训练默认不启用旧合成增强

## 5. 推理改动

在 `inference.py` 中新增：

```text
--dataset kitti_odom
--odom-root D:\kitti_odometry
--seqs 00
--start 0
--count 20
--gap 1
```

推理逻辑：

- 不再扫描扁平 `*.bin` 目录。
- 按 sequence 内帧号构造 pair。
- 对 source/target 分别采样 8192 点。
- 使用 target mean 和 `SCALE_NORM_FACTOR` 归一化。
- 模型输出 `pred_flow` 后乘回 scale。
- 输出文件名必须带 sequence，避免和旧 KITTI detection 结果混淆。

输出命名：

```text
odom_00_000000_to_000001.csv
odom_00_000001_to_000002.csv
```

CSV 列约定：

```text
0:3 = model registered source
3:6 = original source
6:9 = pose-aligned source，可选；有 pose 时写入
```

同时建议每个 prediction 输出一个同名 metrics JSON：

```text
odom_00_000000_to_000001.metrics.json
```

记录：

- source/target 点数
- pose 是否可用
- relative translation / rotation
- original source -> target NN
- pose-aligned source -> target NN
- model-registered source -> target NN
- model registered 与 pose-aligned 的 EPE
- 采样 seed 和参数

## 6. 可视化改动

扩展现有：

```text
visualize_kitti_odometry.py
```

支持读取模型 prediction CSV。

颜色约定：

- 浅蓝灰：source full-scene background
- 深灰：target full-scene background
- 蓝色：original source sample
- 绿色：target sample
- 紫色：pose-aligned source sample
- 橙色：model-registered source sample

输出保持之前 dashboard 风格：

```text
scene_dashboard.html
registration_3d.html
registration_3d.png
registration_3d_focus.png
bev.png
bev_focus.png
full_scene_context_bev.png
trajectory_context.png
displacement_stats.png
metrics.json
```

当前没有图片数据，因此 dashboard 不显示 image overlay 区域。

## 7. 推荐命令

生成 manifest：

```powershell
python -X utf8 D:\Ablation_FiLM\prepare_kitti_odometry.py `
  --odom-root D:\kitti_odometry `
  --out D:\kitti_odometry\prepared\manifests `
  --gap 1
```

debug 训练：

```powershell
python -X utf8 D:\Ablation_FiLM\train.py `
  --dataset kitti_odom `
  --odom-root D:\kitti_odometry `
  --config D:\Ablation_FiLM\config_ppwc_kitti_odom.yaml `
  --debug True `
  --gpu 0
```

正式训练：

```powershell
python -X utf8 D:\Ablation_FiLM\train.py `
  --dataset kitti_odom `
  --odom-root D:\kitti_odometry `
  --config D:\Ablation_FiLM\config_ppwc_kitti_odom.yaml `
  --gpu 0
```

推理：

```powershell
python -X utf8 D:\Ablation_FiLM\inference.py `
  --dataset kitti_odom `
  --odom-root D:\kitti_odometry `
  --seqs 00 `
  --start 0 `
  --count 20 `
  --gap 1 `
  -O D:\predictions\prediction_kitti_odom_supL3\prediction_sup
```

可视化：

```powershell
python -X utf8 D:\Ablation_FiLM\visualize_kitti_odometry.py `
  --odom-root D:\kitti_odometry `
  --prediction-root D:\predictions\prediction_kitti_odom_supL3\prediction_sup `
  --out D:\Ablation_FiLM\visualization_results\kitti_odometry_predictions
```

## 8. 测试计划

数据结构检查：

- `00-21` 的 velodyne/calib 均存在。
- `00-10` 的 pose 行数等于对应 velodyne 帧数。
- manifest 中所有路径存在。

Dataset smoke test：

- 读取 `00:000000 -> 000001`。
- 检查 source/target/color/flow shape 均为 `8192 x 3`。
- 检查 `gt_flow` 非零且无 NaN/Inf。
- 检查 pose-aligned source 到 target 的 NN mean 小于 original source 到 target 的 NN mean。

训练测试：

- `--debug True` 跑至少 1 个 epoch 或少量 batch。
- 检查 loss 可反传。
- 检查 validation EPE 可计算。
- 检查 checkpoint 正常保存。
- 确认 `kitti_odom` 不影响原 `lung` 和旧 `kitti` 分支。

推理测试：

- 先跑 `seq 00` 前 3 对。
- 检查 CSV 存在，列数为 6 或 9，点数为 8192。
- 检查预测无 NaN/Inf。
- 检查输出命名不与旧 KITTI detection 混淆。

可视化测试：

- 对 3 个 odometry prediction case 生成 dashboard。
- 检查 PNG 非空。
- 检查 HTML 引用正常。
- 人眼检查橙色 model prediction、紫色 pose baseline、绿色 target 的关系是否可解释。

## 9. 关键风险

- Odometry pose 只描述 ego-motion，对动态物体不是严格 GT flow。
- 当前模型原始 KITTI 训练逻辑来自单帧合成扰动，直接迁移到真实连续帧可能需要 fine-tune。
- 采样必须固定 seed，否则训练、推理、可视化之间难以对齐。
- `11-21` 无 pose，不应进入 pose-supervised 训练或 pose 指标评估。
- 当前无图像数据，不应继续做 image overlay。

