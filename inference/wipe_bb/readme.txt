wipe_bb / Drifting B-spline：最小本地 Python 部署包
=================================================

模型来源：
  2026.09.16/17.03.02_drifting_bspline_raw_consistency_wipe_bb_wipe_bb_image_bspline/
  checkpoints/epoch=0075-val_loss=8.407.ckpt

这是从上述 checkpoint 导出的 EMA 推理权重，保留 float32 原始精度和全部
归一化参数。原训练 checkpoint 约 4.4 GiB，推理权重约 1.12 GiB。
去掉了优化器状态、另一套非 EMA 权重、训练配置加载流程；没有压缩模型精度。
无需移动原始 4.4 GiB checkpoint、数据集、缓存、训练脚本或整个 bspline_policy。

一、需要移动哪些文件
--------------------
直接移动并解压 wipe_bb_epoch0075.tar.gz，里面只有：

  ema_weights.pt         EMA 权重及训练时的 normalizer；主要体积来源
  metadata.json          网络结构、输入输出、源 checkpoint 哈希及已验证版本
  wipe_bb_policy.py      本地推理加载器，复用已有 diffusion_policy
  self_test.py           离线自检，不导入机械臂驱动、不发送控制指令
  reference_inputs.npz   3 组真实观测、固定噪声与原模型参考输出；仅用于自检
  SHA256SUMS            文件传输完整性校验
  readme.txt            本文件
  LICENSE               代码许可证

日常推理只读取前三个文件。reference_inputs.npz 不是训练数据集，是很小的
验证样例；建议保留完整包，便于迁移后复查。

二、推理机安装与自检
------------------
1. 把压缩包放到推理机，例如 ~/deploy/，解压：

   mkdir -p ~/deploy
   cd ~/deploy
   tar -xzf wipe_bb_epoch0075.tar.gz
   cd wipe_bb_epoch0075
   sha256sum -c SHA256SUMS

2. 激活已经可以运行 Diffusion Policy 的 Python 环境。例如：

   conda activate robodiff

   上面的 robodiff 改为推理机实际环境名。确认导入的是已有 DP：

   python -c "import torch, diffusion_policy; print(torch.__version__); print(torch.cuda.is_available()); print(diffusion_policy.__file__)"

   如果 DP 源码尚未安装到这个环境，设置其外层仓库路径，例如：

   export PYTHONPATH=/path/to/diffusion_policy_repo:$PYTHONPATH

   该路径下应该存在 diffusion_policy/model/ 等目录。不要指向最内层的
   diffusion_policy 包目录；不要用同名的 drifting_policy 分支覆盖已有 DP。

3. 运行离线自检：

   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python self_test.py --device cpu

   有可用 GPU 时，再运行：

   python self_test.py --device cuda:0

   成功结尾为：{"status": "PASS", ..., "robot_commands_sent": 0}。
   自检会核对每个文件 SHA256，严格加载所有权重，用三组真实观测比对原模型
   的 B-spline 参数及最终 7 维动作，并检查原始相机字段与历史帧处理。
   首次加载需要约 2.5 GiB 以上主机内存，实际还需 Python 和运行时开销。
   模型本身约 1.12 GiB；GPU 显存还包含编码器中间结果和 CUDA 工作区。

环境说明：
  加载器使用已有 DP 的 ConditionalUnet1D、normalizer、CropRandomizer
  和 robomimic 编码器；无需安装 bspline_policy、训练 workspace 或 W&B。
  使用 numpy、torch、torchvision、robomimic、scipy、Pillow，以及已有 DP
  自身依赖的 omegaconf、einops、zarr 等。
  仅解码 B-spline，不做拟合，因此本加载器不要求训练时的 scipy>=1.15。

  本次已验证：Python 3.11，torch 2.6.0，torchvision 0.21.0，
  robomimic 0.2.0，numpy 1.26.4，scipy 1.15.2，Pillow 12.3.0。
  完整版本在 metadata.json。推理机其他版本尚未实际验证，以自检结果为准。
  优先沿用已有 CUDA/PyTorch 环境；不需要为本包重新安装训练环境。
  如果缺少 robomimic 且 DP 依赖已齐备：

   python -m pip install robomimic==0.2.0 --no-deps

  不要忽略权重缺失/多余或 shape mismatch 错误；这通常表示 DP/robomimic
  版本或导入路径不同。加载器使用 strict=True，不会悄悄漏载网络参数。

三、接入现有 DP 推理脚本：保留现有采集与控制循环
----------------------------------------------
把包路径加入 Python 搜索路径，只替换原来的 policy 加载代码：

   import sys
   sys.path.insert(0, "/home/你的用户名/deploy/wipe_bb_epoch0075")
   from wipe_bb_policy import WipeBBPolicy

   policy = WipeBBPolicy(
       "/home/你的用户名/deploy/wipe_bb_epoch0075",
       device="cuda:0",
       image_color_order="rgb",
   )

不再用 hydra.utils.get_class(cfg._target_) 创建训练 workspace，也不需要
noise_scheduler 或设置 diffusion inference steps。本模型每次预测只调用一次 UNet。

接口 A：已有 DP 的 obs_dict 张量接口

   result = policy.predict_action(obs_dict)
   actions = result["action"][0].cpu().numpy()   # (8, 7)，单位 rad

obs_dict 必须含以下四项，B 通常为 1，T 固定为 2：

  sideview_image   (B, 2, 3, 128, 128) float32，RGB，范围 [0,1]
  wrist_image      (B, 2, 3, 128, 128) float32，RGB，范围 [0,1]
  ee_pose          (B, 2, 6)           float32，xyz + roll/pitch/yaw
  joint_positions  (B, 2, 7)           float32，当前 7 个关节角，单位 rad

这四项都是未做训练归一化的物理观测，加载器内部自动使用 checkpoint 的
normalizer；图片只先除以 255。不要在外部再次归一化，不要再添加夹爪通道。
图片由 320x240 等原始分辨率直接使用 Pillow LANCZOS 缩放到 128x128，
不要先做 112x112 裁剪；模型内部在 eval 模式做中心裁剪。
image_color_order 参数只作用于下面的原始帧接口；张量接口始终要求 RGB。

接口 B：原始采集帧接口（推荐，自动完成与训练一致的图片预处理）

   old_frame = {
       "D455_color": old_side_rgb,         # uint8，HWC，RGB
       "D405_color": old_wrist_rgb,        # uint8，HWC，RGB
       "ee_pose": old_xyz_rpy,             # (6,)
       "joint_positions": old_joint_q,     # (7,)
   }
   new_frame = {
       "D455_color": new_side_rgb,
       "D405_color": new_wrist_rgb,
       "ee_pose": new_xyz_rpy,
       "joint_positions": new_joint_q,
   }
   actions = policy.predict([old_frame, new_frame])   # numpy (8,7)

  外部相机字段接受 D455_color、D435_color 或 sideview_image；它们代表
  同一外部相机。腕部相机接受 D405_color 或 wrist_image，不可交换相机位置。
  原始图像必须 uint8 HWC；如果采集端输出 OpenCV BGR，初始化时使用
  image_color_order="bgr"，两路统一转换。不要重复 RGB/BGR 交换。
  两帧按旧到新排列，采样间隔约 50 ms，并保持图像和机械臂状态同步。
  ee_pose：位置单位米，欧拉角单位弧度；外旋 xyz，
  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。若采集端只有 4x4 位姿矩阵：

   from scipy.spatial.transform import Rotation
   ee_pose = np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz")]

  joint_positions 是观测；control 是模型应输出的目标值，两者不能互换。

接口 C：每个 20 Hz 采样周期维护历史帧

   policy.reset()  # 每次新 episode/复位后清空历史
   actions = policy.step(frame, infer=need_new_chunk)

  首帧返回 None；之后当 infer=True 返回 (8,7)。正在执行上一个动作块时，
  仍应每个采样周期调用 step(frame, infer=False)，让历史始终是最近连续两帧。
  不要只在每次新动作块开始时塞一帧，否则历史帧间隔会变成约 0.4 秒。
  step 会立即复制采集缓冲区，避免相机重用内存改写历史。

四、输出与现有机械臂控制器的对应
------------------------------
  result["action"] 的形状为 (B,8,7)：
    8 个时间步，每步为 [joint_1_target, ..., joint_7_target]，绝对关节角 rad。
  在现有控制器中按 20 Hz 顺序执行，8 步对应约 0.4 秒；更高频的底层
  插值与关节控制继续由你现有 DP 部署的控制器负责。

  模型没有夹爪动作。第 7 维是第 7 个关节，不能裁成 [-1,1]，不能阈值化
  当作夹爪，也不能补成增量再与当前关节位置相加。
  result["action_pred"] / result["bspline_action"] 是 (B,16,8) 的样条参数，
  不能直接发给机械臂；result["action"] 已经完成 knot 投影与整数时刻解码，
  不需要再次解码、反归一化或切掉第一行。

  加载器本身不发送机械臂指令。把 actions 接到现有的绝对关节目标执行接口，
  保留已有控制器的关节限制、速度限制和停止机制。模型动作未被额外截断。
  首次预热可使用真实采集的两帧调用 predict 并丢弃返回值，再开始正式控制。
  现有真机驱动与采集脚本未随本包提供；它们继续在推理机上使用。

五、本包的验证边界
----------------
  导出时逐个核对全部 EMA 张量，与原 checkpoint 完全相同。
  用 3 组真实观测和相同噪声，与原始完整策略逐项比较 B-spline 参数、
  投影后的参数和最终动作；CPU 上要求逐位一致，结果记录于 metadata.json。
  GPU、相机实采和实际机械臂闭环尚未在目标推理机验证；self_test.py 可在
  目标机分别运行 CPU/GPU 比对。val_loss=8.407 是训练日志指标，不是成功率。

  如果动作或参数参考比对失败，请保留错误信息和自检打印的环境版本，
  先核对 DP 导入路径与模型依赖。不要通过删掉自检断言或 strict=True 绕过。
