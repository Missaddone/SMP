# SMP BodyVelocity LAFAN Walk Posture 实验记录

## 基本信息

- 实验时间：2026-06-23 20:44:10
- 任务名：`Smp-G1-BodyVelocity-LafanWalk-Posture-v0`
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_walk_posture_g1/2026-06-23_20-44-10_smp_body_velocity_lafan_walk_posture_g1`
- 基础代码提交：`3fabb3b Add LAFAN walk-run body velocity task`
- 本次运行包含未提交 diff：`git/SMP_isaaclab.diff`
- 训练方式：从头训练，`resume: false`
- 设备：`cuda:0`
- 环境数：`4096`
- 最大迭代数：`10000`
- policy 频率：50 Hz，`sim.dt=0.005`，`decimation=4`

## 实验目的

在 LAFAN walk local-norm 实验中，GSI/reset 初始状态并没有明显弯腰，但训练后的策略会出现弓背走路。这说明问题更可能来自任务 reward 的局部最优，而不是 prior 或 GSI 初始化本身。

本实验新建独立任务，在当前 walk strong-style 设置基础上加入轻量姿态正则，用来测试显式姿态约束是否能减少弓背走路：

- 约束 base roll/pitch 更直；
- 约束 root 不要塌得太低；
- 只约束腰部 roll/pitch，不约束腿部；
- 加轻微 action rate 平滑。

## Style / SMP 先验

仍然使用 local-normalized LAFAN walk checkpoint：

```text
datasets/pretrain_ckpt/lafan_walk_local_norm.pt
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

主奖励仍为 walk strong-style 的 soft style blending：

```text
task_smp_blend:
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

新增姿态正则：

```text
base_upright:
  weight: -0.2
```

作用：惩罚 base roll/pitch 倾斜，使用 `projected_gravity_b[:2]^2`。

```text
root_height:
  weight: -0.2
  target_height: 0.74
```

作用：只惩罚 root 高度低于目标值，避免为了稳定而塌低重心。

```text
waist_pitch_roll:
  weight: -0.1
  joints:
    - waist_roll_joint
    - waist_pitch_joint
```

作用：直接抑制腰部 roll/pitch 过大。这里没有约束腰 yaw，也没有约束腿部关节，避免压制正常步态。

```text
action_rate:
  weight: -0.005
```

作用：轻微平滑动作，减少高频抖动。

## 和上一版 LAFAN Walk Strong Style 实验的区别

参考实验：

```text
logs/rsl_rl/smp_body_velocity_lafan_walk_g1/2026-06-23_19-38-33_smp_body_velocity_lafan_walk_g1
```

相同点：

- 使用 `lafan_walk_local_norm.pt`。
- command 范围同为 `0.15~1.6 m/s`，yaw `±0.8 rad/s`。
- `style_floor=0.1`。
- actor 不输入 `base_lin_vel`，critic 保留。

不同点：

- 新增 `Smp-G1-BodyVelocity-LafanWalk-Posture-v0` 独立任务。
- 新增 `base_upright`、`root_height`、`waist_pitch_roll`、`action_rate`。
- 不加入脚部正则，先单独验证上身/腰部姿态约束是否能解决弓背。

## 重点观察指标

本次重点观察：

- 弓背走路是否明显减少。
- root 高度是否更稳定，不再塌低。
- 腰部 pitch/roll 是否被压住。
- 速度跟踪是否被姿态正则明显削弱。
- walk 步态是否仍然自然，腿部是否被间接压得不敢动。
- 是否出现为了保持直立而步幅变小、拖步或抖动。

如果弓背减少但速度跟踪变差，可以先降低：

```text
root_height: -0.2 -> -0.1
waist_pitch_roll: -0.1 -> -0.05
```

如果弓背仍明显，可以考虑进一步加入脚部正则或增大腰部约束：

```text
waist_pitch_roll: -0.1 -> -0.15
```
