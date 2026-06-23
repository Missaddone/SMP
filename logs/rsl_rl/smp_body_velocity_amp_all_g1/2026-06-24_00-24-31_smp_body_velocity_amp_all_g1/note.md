# SMP BodyVelocity AMP All 实验记录

## 基本信息

- 日期：2026-06-24 00:24
- 日志目录：`logs/rsl_rl/smp_body_velocity_amp_all_g1/2026-06-24_00-24-31_smp_body_velocity_amp_all_g1`
- 任务：`Smp-G1-BodyVelocity-AmpAll-v0`
- 训练方式：新任务重新训练，不从 checkpoint resume
- 设备：`cuda:0`
- 最大迭代数：`10000`

## 实验目的

本次实验使用新的全量 AMP motion prior：`amp_all.pt`。相比之前的 `lafan_walk_local_norm.pt` 和 `lafan_walk_run_local_nrom.pt`，这个 prior 覆盖的数据范围更广，目标是测试它能否在 BodyVelocity 任务中更好地支持宽速度范围和较大 yaw rate，同时避免单一 walk 或 walk-run prior 中出现的弓背、低速异常步态、风格覆盖不足等问题。

本实验不使用 `style_floor`，reward 使用纯粹的 product 形式：

`reward = task_tracking_reward * style_reward`

这样可以观察全量 prior 本身是否足够覆盖 `0~5 m/s` 的速度命令，而不是依赖 task reward 的常数底座绕过 style 约束。

## Command 设置

- command 类型：机体系 body velocity command
- command 内容：`[v_x_body, v_y_body, yaw_rate]`
- 速度范围：`0.0~5.0 m/s`
- yaw rate 范围：`-2.0~2.0 rad/s`
- `stand_sample_prob=0.0`
- 不单独加入 stand 分支，速度为 0 附近也由同一个 BodyVelocity reward 处理。

## Reward 设置

核心 reward 为 `body_velocity_task_smp_product`：

- 线速度跟踪权重：`lin_vel_weight=0.75`
- yaw rate 跟踪权重：`yaw_rate_weight=0.25`
- `style_floor=0.0`
- `use_stand_branch=False`
- SMP 固定时间步：`fixed_timesteps=(8, 15, 22)`
- SMP 窗口参数：`ws=6.0`
- prior checkpoint：`datasets/pretrain_ckpt/amp_all.pt`

## 相比前面 LAFAN Walk / WalkRun 实验的变化

- prior 从局部 walk 或 walk-run prior 改为全量 motion prior：`amp_all.pt`
- command 速度范围设为 `0.0~5.0 m/s`
- yaw rate 最大值设为 `2.0 rad/s`
- `style_floor=0.0`，不使用 soft blending 常数项
- 不额外加入 posture reward 或 foot regularization，先单独验证全量 prior 的基础效果。

## 观察重点

- 是否比 LAFAN walk / walk-run prior 更少出现弓背走路。
- 低速时是否能形成自然双脚交替，而不是脚尖拖地或单脚支撑。
- 中高速是否能自然进入跑步或快速移动风格。
- `0 m/s` 附近是否能保持位置稳定。
- yaw rate 跟踪在移动中是否有改善。
- 全量 prior 是否因为 motion 分布太宽而导致风格约束变弱、动作混杂或训练不稳定。

## 待补充实验结论

- 
