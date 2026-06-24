# SMP BodyVelocity LAFAN Walk 500 Prior 实验记录

## 基本信息

- 日期：2026-06-24 12:05
- 日志目录：`logs/rsl_rl/smp_body_velocity_lafan_walk_g1/2026-06-24_12-05-48_smp_body_velocity_lafan_walk_g1`
- 任务：`Smp-G1-BodyVelocity-LafanWalk-v0`
- 训练方式：重新开新 log 训练，不从 checkpoint resume
- 设备：`cuda:0`
- 最大迭代数：`10000`

## 实验背景

近期多个 prior 实验出现异常步态或姿态问题，包括弓背、低速脚尖拖地、风格和任务跟踪冲突等。因此当前重点怀疑 prior 训练质量、数据规模、归一化统计或 motion window 处理可能存在问题。

本次实验使用重新训练的小规模 walk prior：`lafan_walk_500.pt`。它相对前面的 `lafan_walk_local_norm.pt` 是一个新的对照，用来判断 prior 的训练数据规模/训练结果变化是否会明显影响 BodyVelocity 策略表现。

## 实验目的

- 验证 `lafan_walk_500.pt` 这个小 pretrain model 是否可用。
- 对比前面的 `lafan_walk_local_norm.pt`，观察异常步态是否减轻或加重。
- 检查较小 walk prior 是否能在中低速范围内提供更清晰的 walking style。
- 为后续排查 prior 训练流程、数据处理和归一化问题提供对照。

## Command 设置

- command 类型：机体系 body velocity command
- command 内容：`[v_x_body, v_y_body, yaw_rate]`
- 速度范围：`0.15~1.6 m/s`
- yaw rate 范围：`-0.8~0.8 rad/s`
- `stand_sample_prob=0.0`
- 不使用单独 stand 分支。

## Reward 设置

主 reward 为 `body_velocity_task_smp_product`：

- 线速度跟踪权重：`lin_vel_weight=0.75`
- yaw rate 跟踪权重：`yaw_rate_weight=0.25`
- `style_floor=0.0`
- `use_stand_branch=False`
- SMP 固定时间步：`fixed_timesteps=(8, 15, 22)`
- SMP 窗口参数：`ws=6.0`
- prior checkpoint：`datasets/pretrain_ckpt/lafan_walk_500.pt`

## 相比上一版 LAFAN Walk 实验的变化

对比 `2026-06-23_22-06-49_smp_body_velocity_lafan_walk_g1`：

- prior 从 `lafan_walk_local_norm.pt` 改为 `lafan_walk_500.pt`
- command 保持不变：速度 `0.15~1.6 m/s`，yaw `±0.8 rad/s`
- reward 保持纯 product：`style_floor=0.0`
- 不加入 posture reward、foot regularization 或其他额外约束，尽量只观察 prior 本身差异。

## 观察重点

- 是否仍然出现弓背走路。
- 低速时是否能形成自然双脚交替。
- 是否出现 prior 训练不足导致的动作抖动、保守不动或风格不清晰。
- 速度和 yaw rate 跟踪是否稳定。
- 如果该 prior 明显更差，需要重点检查小数据/短训练 prior 是否欠拟合；如果更好，则说明前面 prior 的数据分布或训练方式可能确实有问题。

## 待补充实验结论

-
