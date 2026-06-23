# SMP BodyVelocity LAFAN WalkRun Strong Style / Wide Command 实验记录

## 基本信息

- 实验时间：2026-06-23 20:16:30
- 任务名：`Smp-G1-BodyVelocity-LafanWalkRun-v0`
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_walk_run_g1/2026-06-23_20-16-30_smp_body_velocity_lafan_walk_run_g1`
- 基础代码提交：`3fabb3b Add LAFAN walk-run body velocity task`
- 本次运行包含未提交 diff：`git/SMP_isaaclab.diff`
- 训练方式：从头训练，`resume: false`
- 设备：`cuda:0`
- 环境数：`4096`
- 最大迭代数：`10000`
- policy 频率：50 Hz，`sim.dt=0.005`，`decimation=4`

## 实验目的

上一版 WalkRun prior 实验：

```text
logs/rsl_rl/smp_body_velocity_lafan_walk_run_g1/2026-06-23_15-21-52_smp_body_velocity_lafan_walk_run_g1
```

使用 `style_floor=0.5`，command 范围为 `0.25~2.6 m/s` 和 yaw `±0.8 rad/s`。play 后发现与 walk-only 任务类似，也出现弓背走路的问题。

本次实验仿照 walk strong-style 实验的思路：

- 降低 `style_floor`，增强 style prior 约束；
- 大幅扩大 command 范围，测试 walk/run prior 在更高速、更大转向下是否能形成更接近 prior 的运动。

## Style / SMP 先验

仍然使用 walk/run local-normalized checkpoint：

```text
datasets/pretrain_ckpt/lafan_walk_run_local_nrom.pt
```

已知 checkpoint 元信息：

```text
data_dir: datasets/g1_walk_run
norm_stats_file: datasets/g1_walk_run_norm_stats.npz
window_size: 10
feature_dim: 59
num_timesteps: 50
num_noise_samples: 10
d_model: 256
nhead: 4
num_layers: 2
use_ema: true
```

注意：文件名中 `nrom` 是拼写错误，但本次按已有文件名加载，不影响训练。`datasets/` 被 `.gitignore` 忽略，换机器时需要手动同步该 checkpoint。

SMP / GSI 设置：

```text
gsi_buffer_size: 4096
gsi_batch_size: 1024
gsi_refresh: 每 48 s 刷新 1024 个样本
fixed_timesteps: (8, 15, 22)
ws: 6.0
```

## Command 设计

仍然使用 BodyVelocity command：

```text
command = [v_x_body, v_y_body, yaw_rate]
```

本次 command 范围：

```text
speed_min: 0.25 m/s
speed_max: 5.00 m/s
yaw_rate_min: -2.0 rad/s
yaw_rate_max:  2.0 rad/s
resampling_time_range: 3.0-8.0 s
stand_sample_prob: 0.0
```

相对上一版 WalkRun：

```text
speed_max: 2.6 -> 5.0 m/s
yaw_rate: ±0.8 -> ±2.0 rad/s
```

本任务仍不包含 stand。速度下限保持 `0.25 m/s`，避免无 stand 数据的 walk/run prior 与零速度站立目标冲突。

## Observation 设计

Actor 保持实机可部署版本：

```text
actor obs:
- base_ang_vel
- projected_gravity
- joint_pos_rel
- joint_vel_rel
- last_action
- command
```

其中：

- actor 中 `base_lin_vel = null`
- critic 中保留 `base_lin_vel`
- critic 使用 10 帧 history

## Reward 设计

本次主奖励仍为 soft style blending：

```text
task_smp_blend:
  func: body_velocity_task_smp_product
  weight: 1.0
  lin_vel_err_scale: 2.0
  yaw_rate_err_scale: 1.0
  lin_vel_weight: 0.75
  yaw_rate_weight: 0.25
  use_stand_branch: false
  style_floor: 0.1
  fixed_timesteps: (8, 15, 22)
  ws: 6.0
```

等价形式：

```text
reward = task_reward * (0.1 + 0.9 * style_reward)
```

相对上一版 WalkRun：

```text
style_floor: 0.5 -> 0.1
```

这会明显增强 style prior 对策略的约束。当动作不像 walk/run prior 时，task reward 只保留 10% 的基础信号。

## 和上一版 WalkRun 实验的区别

参考实验：

```text
logs/rsl_rl/smp_body_velocity_lafan_walk_run_g1/2026-06-23_15-21-52_smp_body_velocity_lafan_walk_run_g1
```

相同点：

- 使用 `lafan_walk_run_local_nrom.pt`。
- actor 不输入 `base_lin_vel`，critic 保留。
- reward 仍为 `body_velocity_task_smp_product` 的 soft blending。
- 没有 stand branch。
- 没有加入脚部正则。

不同点：

- `style_floor` 从 `0.5` 降到 `0.1`，style 约束更强。
- `speed_max` 从 `2.6 m/s` 增加到 `5.0 m/s`。
- yaw rate 范围从 `±0.8 rad/s` 增加到 `±2.0 rad/s`。

## 重点观察指标

本次重点观察：

- 弓背走路是否减少。
- walk/run prior 是否能在强 style 约束下表现出更自然的上身姿态。
- `2.6~5.0 m/s` 高速区间是否稳定，还是出现不自然跑姿或摔倒。
- yaw rate 扩大到 `±2.0` 后，转向是否稳定。
- 速度跟踪是否因为 `style_floor=0.1` 明显变差。
- 是否只学到跑步，低速 walk 是否被破坏。

如果弓背仍然明显，说明仅调 style_floor 可能不够，需要显式加入：

```text
torso/base upright reward
root height reward
waist pitch/roll penalty
foot regularization
```

如果速度/yaw 跟踪明显退化，可以尝试中间值：

```text
style_floor: 0.1 -> 0.3
```
