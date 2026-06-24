# SMP BodyVelocity AMP Run Prior 可行性实验记录

## 基本信息

- 日期：2026-06-24 19:21
- 日志目录：`logs/rsl_rl/smp_body_velocity_amp_run_g1/2026-06-24_19-21-55_smp_body_velocity_amp_run_g1`
- 任务：`Smp-G1-BodyVelocity-AmpRun-v0`
- 训练方式：新任务重新训练，不从 checkpoint resume
- 设备：`cuda:0`
- 最大迭代数：`10000`

## 实验背景

前面已经分别测试了 LAFAN run prior、LAFAN walk prior、LAFAN walk-run prior 和 AMP all prior。由于近期策略仍然出现异常步态、弓背、低速不稳定等问题，目前仍然需要继续排查 prior 训练质量和 motion 数据分布是否是主要原因。

本次实验使用新训练的 `amp_run.pt`，只替换普通 BodyVelocity 任务中的 prior，不额外加入姿态正则或脚部正则，目的是单独验证 AMP run prior 本身是否适合 BodyVelocity 任务。

## 实验目的

- 验证 `amp_run.pt` 作为 SMP prior 是否可用。
- 与 `lafan_run.pt` 做直接对照，判断 AMP run 数据/训练方式是否更稳定。
- 观察 run prior 是否能改善近期出现的异常步态和风格不匹配问题。
- 为后续决定是否继续使用 AMP 数据训练 prior 提供依据。

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
- prior checkpoint：`datasets/pretrain_ckpt/amp_run.pt`

## 相比 LAFAN Run 实验的变化

对比 `2026-06-24_11-37-08_smp_body_velocity_lafan_run_g1`：

- prior 从 `lafan_run.pt` 改为 `amp_run.pt`
- command、reward、observation 均保持普通 BodyVelocity 设置
- 不加入额外 reward 项，尽量只观察 prior 本身差异

## 观察重点

- 是否比 `lafan_run.pt` 更稳定。
- 是否减少弓背、脚尖拖地或低速异常支撑。
- 速度 `0~3 m/s` 范围内是否能形成合理移动风格。
- yaw rate 跟踪是否有改善。
- 如果结果仍然异常，需要继续检查 pretrain 数据处理、归一化统计、GSI 输出和 `task * style` reward 形式。

## 待补充实验结论

-
