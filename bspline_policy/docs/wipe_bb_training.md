# wipe_bb：Drifting B-spline raw concat 训练

训练配置：`train_drifting_unet_wipe_bb_image_bspline_raw_concat_workspace`。

## 数据与动作表示

默认读取 `diffusion_policy/data/wipe_bb/data_save.zarr`，共 67 条轨迹、
23,673 帧，时间戳中位间隔约 0.05009 秒（20 Hz）。

- 观测：两帧双相机 RGB（128×128）、`ee_pose`（6 维）、`joint_positions`（7 维）。
  `D435_color` 实际对应 D455 外部相机，`D405_color` 对应腕部相机。
- 动作：`control` 的 7 维绝对关节目标，不包含夹爪，不做差分或额外时间平移。
- 数据标签：`(16, 15)`，排列为 `[knot, 7D control points, 7D raw actions]`。
- `shape_meta.action.shape: [7]` 表示物理动作维数；模型内部自动添加 knot 通道。
- 沿用仓库当前 raw concat 的 `decode_consistency` 模式：UNet 输出 `(16, 8)`，
  Drifting 损失作用于 B-spline 参数；解码后的动作与原始动作计算权重 0.1 的一致性损失。
  原始动作按样条有效区间的 16 个相位点插值，限制在同一条轨迹内。
- 推理保留 Franka 的整数控制时刻解码，输出 `(8, 7)`，约执行 0.4 秒。
  训练时覆盖整段样条的一致性采样与推理时按控制周期执行的采样用途不同。

默认 seed 为 42，按轨迹划分 60 条训练 / 7 条验证，分别有 21,122 / 2,484
个样本。动作和低维观测归一化只使用训练轨迹；验证集共享训练归一化参数。
样条缓存位于 `bspline_policy/data/cache/wipe_bb/`，首次运行生成，后续复用。
替换数据内容后应更换 `task.dataset.cache_base_path`，避免使用旧缓存。

## 启动训练

```bash
conda activate bsp-simple
cd /home/zmw/bspline-policy/bspline_policy

python train.py \
  --config-name=train_drifting_unet_wipe_bb_image_bspline_raw_concat_workspace \
  training.resume=false
```

默认使用 `cuda:0`，300 epochs，batch size 64，每条观测生成 8 个候选，
离线记录 W&B 日志，以 `val_loss` 保存 checkpoint。任务使用空环境 runner，
训练期间不执行真机动作。

显存不足时可降低 batch size，保持 `training.gradient_accumulate_every=1`：

```bash
python train.py \
  --config-name=train_drifting_unet_wipe_bb_image_bspline_raw_concat_workspace \
  training.resume=false dataloader.batch_size=8 val_dataloader.batch_size=8
```

若只使用双相机和 `ee_pose`，须同时关闭数据读取与模型观测项：

```bash
python train.py \
  --config-name=train_drifting_unet_wipe_bb_image_bspline_raw_concat_workspace \
  training.resume=false task.dataset.include_joint_positions=false \
  '~task.shape_meta.obs.joint_positions'
```

输出目录为 `bspline_policy/data/outputs/<日期>/<时间>_drifting_bspline_raw_consistency_wipe_bb_wipe_bb_image_bspline/`。

## 本次验证

已在 `bsp-simple` 环境检查真实数据的形状、有限值、归一化往返、完整训练动作标签
和 train/val 缓存。使用训练中的 knot 投影和可微解码抽查 256 个样本，最大关节
重建误差约 0.00982 rad。

当前环境 CUDA 不可用；CPU 冒烟测试保留双相机编码器和 8 个候选，将 UNet
缩小为 `[32, 64, 128]`，完成 2 个训练 batch、2 个验证 batch、EMA 更新、
推理解码和 checkpoint 保存。这验证训练链路，不代表完整模型已完成训练或 GPU 显存已验证。

从 `bspline_policy/` 复现冒烟测试：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled python train.py \
  --config-name=train_drifting_unet_wipe_bb_image_bspline_raw_concat_workspace \
  training.device=cpu training.resume=false training.num_epochs=1 \
  training.max_train_steps=2 training.max_val_steps=2 \
  dataloader.batch_size=2 val_dataloader.batch_size=2 \
  dataloader.num_workers=0 val_dataloader.num_workers=0 \
  dataloader.pin_memory=false val_dataloader.pin_memory=false \
  'policy.down_dims=[32,64,128]' policy.diffusion_step_embed_dim=32 \
  logging.mode=disabled logging.resume=false \
  hydra.run.dir=/tmp/wipe_bb_raw_concat_smoke
```

适配回归测试覆盖无夹爪 / 有夹爪数据、区间插值、轨迹边界、训练统计隔离、
缓存复用、双相机损失反传及 7 维推理输出。
