# SMP BodyVelocity LAFAN WalkRun 纯 style product 实验记录

## 基本信息

- 日期：2026-06-23 22:09
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_walk_run_g1/2026-06-23_22-09-00_smp_body_velocity_lafan_walk_run_g1`
- 任务：`Smp-G1-BodyVelocity-LafanWalkRun-v0`
- 训练方式：重新开新 log 训练，不从 checkpoint resume
- 设备：`cuda:0`
- 最大迭代数：`10000`

## 实验目的

上一组 LAFAN WalkRun strong-style 实验在扩大 command 范围的同时将 `style_floor` 从 `0.5` 降到 `0.1`，但仍然观察到弓背走路的问题。本次将 `style_floor` 进一步设为 `0.0`，测试完全依赖 style prior 约束时，walk/run 混合 prior 是否能减少异常躯干姿态，并更好地区分低速 walk 和高速 run。

本实验也是对 walk 纯 style product 实验的配套对照：walk 只覆盖低速行走风格，而 walk-run prior 覆盖更宽速度范围，理论上在 `0.25~5.0 m/s` 的 command 下应更容易给出合理步态。

## Command 设置

- command 类型：机体系 body velocity command
- command 内容：`[v_x_body, v_y_body, yaw_rate]`
- 速度范围：`0.25~5.0 m/s`
- yaw rate 范围：`-2.0~2.0 rad/s`
- `stand_sample_prob=0.0`
- 本实验不单独加入 stand 分支。

## Reward 设置

核心 reward 仍为 `body_velocity_task_smp_product`：

- 线速度跟踪权重：`lin_vel_weight=0.75`
- yaw rate 跟踪权重：`yaw_rate_weight=0.25`
- `style_floor=0.0`
- `use_stand_branch=False`
- SMP 固定时间步：`fixed_timesteps=(8, 15, 22)`
- SMP 窗口参数：`ws=6.0`
- prior checkpoint：`datasets/pretrain_ckpt/lafan_walk_run_local_nrom.pt`

## 相比上一次 LAFAN WalkRun 实验的变化

对比 `2026-06-23_20-16-30_smp_body_velocity_lafan_walk_run_g1`：

- `style_floor: 0.1 -> 0.0`
- command 范围保持不变：速度 `0.25~5.0 m/s`，yaw `±2.0 rad/s`
- prior 保持不变：`lafan_walk_run_local_nrom.pt`
- 不加入额外 posture reward，只单独测试纯 style product 对 walk/run 风格还原和弓背问题的影响。

## 观察重点

- 是否减少弓背走路。
- 低速是否能形成正常 walk，而不是脚尖拖地或单脚支撑。
- 中高速是否能自然切换到 run 风格。
- 速度和 yaw rate 跟踪是否因为 `style_floor=0.0` 而明显变差。
- 是否出现 style prior 过强导致的保守运动、抖动或训练不稳定。

## 待补充实验结论

- 
