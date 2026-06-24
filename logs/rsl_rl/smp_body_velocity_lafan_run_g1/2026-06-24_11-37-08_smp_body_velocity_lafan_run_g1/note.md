# SMP BodyVelocity LAFAN Run Prior 可行性实验记录

## 基本信息

- 日期：2026-06-24 11:37
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_run_g1/2026-06-24_11-37-08_smp_body_velocity_lafan_run_g1`
- 任务：`Smp-G1-BodyVelocity-LafanRun-v0`
- 训练方式：新任务重新训练，不从 checkpoint resume
- 设备：`cuda:0`
- 最大迭代数：`10000`

## 实验背景

最近多组使用 LAFAN walk、LAFAN walk-run、AMP all prior 的 BodyVelocity 实验都出现了一些不理想现象，例如弓背、低速步态异常、脚尖拖地、风格和任务跟踪之间冲突等。因此目前怀疑 prior 本身的训练或数据处理可能存在较大问题。

本次实验不加入额外姿态正则、不扩大复杂 reward，也不使用新的混合 prior，而是使用一个更直接的 LAFAN run prior：`lafan_run.pt`，在普通 BodyVelocity 任务上单独验证该 prior 是否可用。

## 实验目的

- 单独测试 `lafan_run.pt` 作为 SMP prior 时，是否能支持普通 BodyVelocity 训练。
- 判断异常步态是否来自特定 prior，例如 walk / walk-run / amp_all，而不是 BodyVelocity 任务本身。
- 观察 run prior 是否比 walk prior 更适合 `0~3 m/s` 的移动 command。
- 为后续排查 prior 训练流程、motion 数据归一化、GSI 初始化和 reward product 形式提供对照。

## Command 设置

- command 类型：机体系 body velocity command
- command 内容：`[v_x_body, v_y_body, yaw_rate]`
- 速度范围：`0.0~3.0 m/s`
- yaw rate 范围：`-2.0~2.0 rad/s`
- `stand_sample_prob=0.0`
- 不使用单独 stand 分支。

## Reward 设置

继承普通 `Smp-G1-BodyVelocity-v0` 的 reward：

- reward 函数：`body_velocity_task_smp_product`
- 线速度跟踪权重：`lin_vel_weight=0.75`
- yaw rate 跟踪权重：`yaw_rate_weight=0.25`
- `use_stand_branch=False`
- SMP 固定时间步：`fixed_timesteps=(8, 15, 22)`
- SMP 窗口参数：`ws=6.0`
- prior checkpoint：`datasets/pretrain_ckpt/lafan_run.pt`

## 相比普通 BodyVelocity 任务的变化

- 主要变化只有 prior：使用 `lafan_run.pt`。
- command、reward、observation 基本沿用普通 BodyVelocity。
- actor 仍不使用实机不可获取的 `base_lin_vel_b`，critic 保留对应信息。
- 不加入 posture reward、foot regularization、hand collision penalty 或 style floor 调整。

## 观察重点

- 机器人是否能形成稳定的跑步/移动风格。
- 是否仍然出现明显弓背，如果出现，说明问题可能不只在 walk prior。
- 低速和 `0 m/s` 附近是否稳定，是否会漂移或抖动。
- yaw rate 跟踪是否比之前更好。
- 是否出现先验风格过强导致任务跟踪困难，或 prior 过弱导致动作混乱。
- 如果训练结果仍然异常，需要进一步检查 prior 训练数据、归一化统计、window 采样、GSI 输出和 `task * style` reward 形式。

## 待补充实验结论

-
