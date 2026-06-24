# SMP BodyVelocity LAFAN Walk Posture 宽 command 实验记录

## 基本信息

- 日期：2026-06-24 02:08
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_walk_posture_g1/2026-06-24_02-08-57_smp_body_velocity_lafan_walk_posture_g1`
- 任务：`Smp-G1-BodyVelocity-LafanWalk-Posture-v0`
- 训练方式：重新开新 log 训练，不从 checkpoint resume
- 设备：`cuda:0`
- 最大迭代数：`10000`

## 实验目的

前一版 posture 实验主要验证显式姿态正则是否能缓解 LAFAN walk prior 下的弓背问题，command 仍然保持在较窄的 walk 范围内。本次将同一个 posture 任务的 command 扩大到速度 `0~3 m/s`、yaw `±2 rad/s`，用于测试：

- 姿态正则在更宽速度和转向范围下是否仍然能抑制弓背。
- LAFAN walk prior 在更高速度 command 下是否会出现风格不匹配、拖脚或抖动。
- 宽 command 下速度/yaw 跟踪和身体姿态约束之间的权衡。

## Command 设置

- command 类型：机体系 body velocity command
- command 内容：`[v_x_body, v_y_body, yaw_rate]`
- 速度范围：`0.0~3.0 m/s`
- yaw rate 范围：`-2.0~2.0 rad/s`
- `stand_sample_prob=0.0`
- 不单独加入 stand 分支。

## Reward 设置

主 reward 为 `body_velocity_task_smp_product`：

- 线速度跟踪权重：`lin_vel_weight=0.75`
- yaw rate 跟踪权重：`yaw_rate_weight=0.25`
- `style_floor=0.0`
- `use_stand_branch=False`
- SMP 固定时间步：`fixed_timesteps=(8, 15, 22)`
- SMP 窗口参数：`ws=6.0`
- prior checkpoint：`datasets/pretrain_ckpt/lafan_walk_local_norm.pt`

额外姿态正则：

- `base_upright`：权重 `-0.2`，惩罚 base roll/pitch 倾斜。
- `root_height`：权重 `-0.2`，`target_height=0.74`，只惩罚 root 低于目标高度。
- `waist_pitch_roll`：权重 `-0.1`，约束 `waist_roll_joint` 和 `waist_pitch_joint` 靠近 0。
- `action_rate`：权重 `-0.005`，抑制动作抖动。

## 相比上一版 posture 实验的变化

对比 `2026-06-23_20-44-10_smp_body_velocity_lafan_walk_posture_g1`：

- 速度范围：`0.15~1.6 m/s -> 0.0~3.0 m/s`
- yaw rate 范围：`±0.8 rad/s -> ±2.0 rad/s`
- 姿态正则项保持不变。
- prior 保持 `lafan_walk_local_norm.pt`。
- `style_floor=0.0`，继续使用纯 `task * style` product。

## 观察重点

- 宽 command 后是否重新出现弓背，或者 posture reward 仍然有效。
- 高速 command 下是否因 walk prior 覆盖不足而出现不自然移动。
- 低速和 0 速附近是否能保持稳定，不明显漂移。
- yaw rate 跟踪是否改善，还是受姿态正则/先验限制。
- 是否出现 root height reward 过强导致动作僵硬。

## 待补充实验结论

-
