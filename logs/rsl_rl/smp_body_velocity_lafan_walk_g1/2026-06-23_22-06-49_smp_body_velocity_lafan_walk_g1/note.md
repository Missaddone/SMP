# SMP BodyVelocity LAFAN Walk 纯 style product 实验记录

## 基本信息

- 日期：2026-06-23 22:06
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_walk_g1/2026-06-23_22-06-49_smp_body_velocity_lafan_walk_g1`
- 任务：`Smp-G1-BodyVelocity-LafanWalk-v0`
- 训练方式：重新开新 log 训练，不从 checkpoint resume
- 设备：`cuda:0`
- 最大迭代数：`10000`

## 实验目的

上一组 LAFAN Walk strong-style 实验将 `style_floor` 从 `0.5` 降到 `0.1`，希望增强先验约束并缓解弓背走路问题。本次进一步将 `style_floor` 直接设为 `0.0`，把 reward 变成更纯粹的 product 形式：

`reward = task_tracking_reward * style_reward`

这样做的目的是观察：当 task reward 完全不能绕过 style prior 时，机器人是否会更接近 LAFAN walk prior 的身体姿态和步态，而不是通过弓背等异常姿态去完成速度跟踪。

## Command 设置

- command 类型：机体系 body velocity command
- command 内容：`[v_x_body, v_y_body, yaw_rate]`
- 速度范围：`0.15~1.6 m/s`
- yaw rate 范围：`-0.8~0.8 rad/s`
- `stand_sample_prob=0.0`
- 本实验不单独加入 stand 分支，低速仍作为 walk 任务处理。

## Reward 设置

核心 reward 仍为 `body_velocity_task_smp_product`：

- 线速度跟踪权重：`lin_vel_weight=0.75`
- yaw rate 跟踪权重：`yaw_rate_weight=0.25`
- `style_floor=0.0`
- `use_stand_branch=False`
- SMP 固定时间步：`fixed_timesteps=(8, 15, 22)`
- SMP 窗口参数：`ws=6.0`
- prior checkpoint：`datasets/pretrain_ckpt/lafan_walk_local_norm.pt`

## 相比上一次 LAFAN Walk 实验的变化

对比 `2026-06-23_19-38-33_smp_body_velocity_lafan_walk_g1`：

- `style_floor: 0.1 -> 0.0`
- command 范围保持不变：速度 `0.15~1.6 m/s`，yaw `±0.8 rad/s`
- prior 保持不变：`lafan_walk_local_norm.pt`
- 不加入额外 posture reward，只单独测试纯 style product 对弓背问题的影响。

## 观察重点

- 是否明显减少弓背走路。
- 低速时是否更接近正常 walk，而不是脚尖拖地或单脚异常支撑。
- 速度跟踪是否因为 `style_floor=0.0` 而明显变差。
- yaw rate 跟踪是否变得更难。
- 是否出现 prior 过强导致策略不愿意运动、原地抖动或训练不稳定。

## 待补充实验结论

- 
