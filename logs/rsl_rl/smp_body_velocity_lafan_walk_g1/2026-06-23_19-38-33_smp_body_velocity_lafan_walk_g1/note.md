# SMP BodyVelocity LAFAN Walk Strong Style 实验记录

## 基本信息

- 实验时间：2026-06-23 19:38:33
- 任务名：`Smp-G1-BodyVelocity-LafanWalk-v0`
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_walk_g1/2026-06-23_19-38-33_smp_body_velocity_lafan_walk_g1`
- 基础代码提交：`3fabb3b Add LAFAN walk-run body velocity task`
- 本次运行包含未提交 diff：`git/SMP_isaaclab.diff`
- 训练方式：从头训练，`resume: false`
- 设备：`cuda:0`
- 环境数：`4096`
- 最大迭代数：`10000`
- policy 频率：50 Hz，`sim.dt=0.005`，`decimation=4`

## 实验目的

上一版 local-norm LAFAN walk 实验：

```text
logs/rsl_rl/smp_body_velocity_lafan_walk_g1/2026-06-23_11-57-07_smp_body_velocity_lafan_walk_g1
```

使用 `style_floor=0.5`，task 信号较强。play 时观察到机器人可能出现弓背走路，说明当前 reward 对上身姿态和 walk 风格的约束可能不够强。

本次实验在同一个 local-norm walk prior 基础上：

- 降低 `style_floor`，增强 style prior 对动作的约束；
- 同时稍微扩大 command 范围，测试在更强风格约束下能否覆盖更大速度和转向。

## Style / SMP 先验

仍然使用 local-normalized LAFAN walk checkpoint：

```text
datasets/pretrain_ckpt/lafan_walk_local_norm.pt
```

已知 checkpoint 元信息：

```text
data_dir: datasets/g1_walk
norm_stats_file: datasets/lafan_walk_norm_stats.npz
window_size: 10
```

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
speed_min: 0.15 m/s
speed_max: 1.60 m/s
yaw_rate_min: -0.8 rad/s
yaw_rate_max:  0.8 rad/s
resampling_time_range: 3.0-8.0 s
stand_sample_prob: 0.0
```

相对上一版：

```text
speed_max: 1.2 -> 1.6 m/s
yaw_rate:  ±0.5 -> ±0.8 rad/s
```

本任务仍不包含 stand。速度下限保持 `0.15 m/s`，避免 walk prior 与零速度站立目标冲突。

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

相对上一版：

```text
style_floor: 0.5 -> 0.1
```

这意味着 style reward 的影响明显增强。当动作不像 walk prior 时，task reward 最多只保留 10% 的基础信号，而不是上一版的 50%。

## 和上一版 LAFAN Walk Local Norm 实验的区别

参考实验：

```text
logs/rsl_rl/smp_body_velocity_lafan_walk_g1/2026-06-23_11-57-07_smp_body_velocity_lafan_walk_g1
```

相同点：

- 使用 `lafan_walk_local_norm.pt`。
- actor 不输入 `base_lin_vel`，critic 保留。
- reward 仍为 `body_velocity_task_smp_product` 的 soft blending。
- 没有 stand branch。

不同点：

- `style_floor` 从 `0.5` 降到 `0.1`，style 约束更强。
- `speed_max` 从 `1.2 m/s` 增加到 `1.6 m/s`。
- yaw rate 范围从 `±0.5 rad/s` 增加到 `±0.8 rad/s`。

## 重点观察指标

本次主要观察：

- 弓背走路是否减少。
- 上身姿态是否更接近 prior 中的 walk。
- 速度跟踪是否因为 style 变强而明显变差。
- `0.15-0.5 m/s` 低速是否仍然能稳定前进。
- `1.2-1.6 m/s` 区间是否能维持自然 walk/快走，而不是抖腿或拖滑。
- yaw rate 扩大到 `±0.8` 后，转向是否稳定。

如果机器人更像 walk 但速度跟踪下降严重，说明 `style_floor=0.1` 可能过低，可以尝试中间值：

```text
style_floor: 0.1 -> 0.3
```

如果弓背仍然明显，说明问题可能不是 style_floor 不够，而是需要显式加入：

```text
torso/base upright reward
root height reward
waist pitch/roll penalty
foot regularization
```
