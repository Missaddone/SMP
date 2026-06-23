# SMP IsaacLab 实验记录汇总

更新时间：2026-06-23

本文档根据 `logs/rsl_rl/**/note.md` 汇总，用于快速查看各次实验的日期、任务名称和主要修改。实验结论列预留给后续人工补充。

## 阶段概览

| 阶段 | 核心问题 | 主要实验方向 |
|---|---|---|
| 6月14日-6月18日 | 原 steering/modified 任务中低速、站立、face reward、deadzone 设计不稳定 | 反复调整低速采样、deadzone、站立 reward、face reward 是否参与 |
| 6月19日 | 开始转向更适合实机部署的机体系 command | 创建 `Steering-with-stand`、`BodyVelocity`、`ZeroVelocity` 三类任务 |
| 6月20日 | 实机 actor 不应依赖 `base_lin_vel` | 改成 asymmetric actor-critic：actor 去掉 `base_lin_vel`，critic 保留 |
| 6月20日-6月21日 | prior 和 command 范围对运动效果影响明显 | 比较 `lafan_run`、`loco` prior，并扩大 yaw-rate 范围 |
| 6月22日-6月23日 | 低速步态不自然、walk prior 不匹配 | 尝试 LAFAN walk prior、本地归一化 prior、收窄 command、soft style blending |
| 6月23日 | 低速单脚脚尖着地、左右脚不交替 | 新建 FootRegularized 任务，加入脚部接触/倾斜/滑动/单支撑惩罚 |

## 实验表

| 日期时间 | 实验名称 / 日志目录 | 任务 | 主要修改 / 实验目的 | 实验结论 |
|---|---|---|---|---|
| 2026-06-14 03:10 | `smp_steering_modified_g1/2026-06-14_03-10-36` | Steering Modified | 速度采样 `0-3.0 m/s`；`0-0.5 m/s` 作为 deadzone，reward 中参考速度视作 0；加入站立额外约束：root 速度、关节速度、action 惩罚。 |  |
| 2026-06-14 16:10 | `smp_steering_modified_g1/2026-06-14_16-10-47` | Steering Modified | 在上一版基础上，将 `0-0.5 m/s` 低速采样概率提高到 `40%`；提高 velocity 权重到 `1.5`；加大 joint/action penalty。 |  |
| 2026-06-14 21:49 | `smp_steering_modified_g1/2026-06-14_21-49-15` | Steering Modified | 低速采样概率 `40%`；deadzone reward 中参考速度加入 `deadzone_bias=-0.05`，不再严格按 0 处理。 |  |
| 2026-06-15 01:59 | `smp_steering_modified_g1/2026-06-15_01-59-48` | Steering Modified | 将重点低速区间改为 `0-0.1 m/s`，采样概率 `30%`；deadzone bias `-0.05`。 |  |
| 2026-06-15 02:03 | `smp_steering_modified_g1/2026-06-15_02-03-29` | Steering Modified | 回到 `0-0.5 m/s` 低速区间，采样概率 `40%`；deadzone bias 改为 `-0.1`。 |  |
| 2026-06-15 11:26 | `smp_steering_modified_g1/2026-06-15_11-26-43` | Steering Modified | `0-0.1 m/s` 低速采样概率提高到 `40%`；deadzone bias `-0.1`。 |  |
| 2026-06-15 11:40 | `smp_steering_modified_g1/2026-06-15_11-40-24` | Steering Modified | 速度采样改为 `-0.2~3.0 m/s`；死区不启用，速度范围内均匀采样；保留 deadzone 相关 reward 结构。 |  |
| 2026-06-16 01:42 | `smp_steering_modified_g1/2026-06-16_01-42-25` | Steering Modified | 原 SMP 工程修改版；使用 `lafan1` 数据；站立使用普通 reward，移动使用 SMP reward；速度采样 `-0.5~5.0`，`speed<0` 为死区。 |  |
| 2026-06-16 14:51 | `smp_steering_modified_g1/2026-06-16_14-51-15` | Steering Modified | 同上一版：`lafan1`，站立普通 reward，移动 SMP reward，采样 `-0.5~5.0`；因中断重新训练。 |  |
| 2026-06-16 15:25 | `smp_steering_modified_g1/2026-06-16_15-25-53` | Steering Modified / Stand Test | 速度采样 `-0.5~0.1`，`speed<0.1` 视为死区，基本是纯站立训练；用于判断站立姿态问题是否由 moving 影响。 |  |
| 2026-06-16 23:42 | `smp_steering_modified_g1/2026-06-16_23-42-08` | Steering Modified / Stand Test | 仍为纯站立测试；因站立姿态不对，重新调整站立 reward。 |  |
| 2026-06-17 18:06 | `smp_steering_modified_g1/2026-06-17_18-06-31` | Steering Modified / Stand Test | 继续测试站立 reward。note 中记录：依旧不能很好站立，怀疑 face reward 有影响，参数未调好。 |  |
| 2026-06-17 19:50 | `smp_steering_modified_g1/2026-06-17_19-50-20` | Steering Modified / Stand Test | 重新设计 reward 和权重；站立时不考虑 face reward。 |  |
| 2026-06-18 00:16 | `smp_steering_modified_g1/2026-06-18_00-16-04` | Steering Modified | 速度采样 `-0.5~5.0`，`-0.1<speed<0.1` 为死区；站立不考虑 face reward 后已能正常站立，因此重新开启 moving 任务。 |  |
| 2026-06-18 11:40 | `smp_steering_modified_g1/2026-06-18_11-40-52` | Steering Modified | 在上一版基础上减少死区采样频率到 `0.2`，同时降低最大速度；note 中主观判断“可能不会有用”。 |  |
| 2026-06-19 02:48 | `smp_steering_with_stand_g1/2026-06-19_02-48-18_smp_steering_with_stand_g1` | `Smp-G1-Steering-with-stand-v0` | 新建机体系 command `[v_x_body, v_y_body, yaw_rate]` 的 steering-with-stand 任务；moving 使用 `(velocity+yaw tracking)*SMP`，stand 使用 `standing_pose_reward`；stand 采样概率 `0.2`。本次 run 手动终止，仅保留启动记录。 |  |
| 2026-06-19 02:54 | `smp_body_velocity_g1/2026-06-19_02-54-51_smp_body_velocity_g1` | `Smp-G1-BodyVelocity-v0` | 创建/验证无 stand 的 BodyVelocity 任务；command 为 `[v_x_body, v_y_body, yaw_rate]`；速度 `0.5~3.0`，yaw `±1.0`；使用 `pretrained_lafan_run.pt`；reward 为 task tracking 乘 SMP。 |  |
| 2026-06-19 15:09 | `smp_zero_velocity_g1/2026-06-19_15-09-14_smp_zero_velocity_g1` | `Smp-G1-ZeroVelocity-v0` | 创建零速度站立任务；command 固定 `[0,0,0]`；使用站立 prior `pretrained_jushen_stand.pt`；不使用显式 `standing_pose_reward`，而是零速度 tracking 乘 standing SMP。 |  |
| 2026-06-20 14:34 | `smp_body_velocity_g1/2026-06-20_14-34-02_smp_body_velocity_g1` | `Smp-G1-BodyVelocity-v0` | Actor 移除不可实机获取的 `base_lin_vel_b`，输入从 99 维变 96 维；critic 保留 `base_lin_vel` 和 10 帧 history；command/reward/prior 不变。 |  |
| 2026-06-20 15:56 | `smp_body_velocity_g1/2026-06-20_15-56-51_smp_body_velocity_g1` | `Smp-G1-BodyVelocity-v0` | 保持 BodyVelocity、obs、command、reward 不变，仅将 prior 从 `pretrained_lafan_run.pt` 改为 `pretrained_loco.pt`；用于比较不同 locomotion prior。 |  |
| 2026-06-21 20:06 | `smp_body_velocity_g1/2026-06-21_20-06-35_smp_body_velocity_g1` | `Smp-G1-BodyVelocity-v0` | 扩展 BodyVelocity command：速度 `0.5~3.0 -> 0.0~3.0`，yaw `±1.0 -> ±2.0`；prior 从 `pretrained_loco.pt` 回到 `pretrained_lafan_run.pt`；actor 仍无 `base_lin_vel`。 |  |
| 2026-06-22 16:04 | `smp_body_velocity_lafan_walk_g1/2026-06-22_16-04-32_smp_body_velocity_lafan_walk_g1` | `Smp-G1-BodyVelocity-LafanWalk-v0` | 使用 `pretrained_lafan_walk.pt` 替换 `pretrained_lafan_run.pt`；其余 BodyVelocity 设置基本沿用 6-21：速度 `0~3.0`，yaw `±2.0`；用于测试 walk prior 是否改善低速步态。 |  |
| 2026-06-23 11:45 | `smp_body_velocity_foot_regularized_g1/2026-06-23_11-45-25_smp_body_velocity_foot_regularized_g1` | `Smp-G1-BodyVelocity-FootRegularized-v0` | 在原 BodyVelocity + `pretrained_lafan_run.pt` 上加入脚部正则：`feet_air_time=0.25`、`feet_slide=-0.1`、`support_foot_tilt=-0.1`、`persistent_single_support=-0.2`、`action_rate=-0.005`；用于解决低速单脚脚尖着地和不交替。 |  |
| 2026-06-23 11:57 | `smp_body_velocity_lafan_walk_g1/2026-06-23_11-57-07_smp_body_velocity_lafan_walk_g1` | `Smp-G1-BodyVelocity-LafanWalk-v0` | 使用新的 local norm walk prior：`lafan_walk_local_norm.pt`；command 收窄到 walk 合理范围：速度 `0.15~1.2`，yaw `±0.5`；reward 改为 `style_floor=0.5` 的 soft blending，避免 task 信号消失。 |  |
| 2026-06-23 15:21 | `smp_body_velocity_lafan_walk_run_g1/2026-06-23_15-21-52_smp_body_velocity_lafan_walk_run_g1` | `Smp-G1-BodyVelocity-LafanWalkRun-v0` | 使用新的 walk/run local norm prior：`lafan_walk_run_local_nrom.pt`；command 设置为速度 `0.25~2.6 m/s`、yaw `±0.8 rad/s`；reward 使用 `style_floor=0.5` 的 soft blending；第一版不加入脚部正则，用于单独验证 walk/run prior 是否能同时还原低速 walk、中高速 run 和速度过渡。 |  |
| 2026-06-23 19:38 | `smp_body_velocity_lafan_walk_g1/2026-06-23_19-38-33_smp_body_velocity_lafan_walk_g1` | `Smp-G1-BodyVelocity-LafanWalk-v0` | 在 local norm walk prior 实验基础上增强 style 约束：`style_floor 0.5 -> 0.1`；同时扩大 command：速度 `0.15~1.2 -> 0.15~1.6 m/s`，yaw `±0.5 -> ±0.8 rad/s`；用于观察更强 walk prior 是否能减少弓背/异常姿态，同时测试稍大速度和转向范围。 |  |
| 2026-06-23 20:16 | `smp_body_velocity_lafan_walk_run_g1/2026-06-23_20-16-30_smp_body_velocity_lafan_walk_run_g1` | `Smp-G1-BodyVelocity-LafanWalkRun-v0` | 在 WalkRun local norm prior 实验基础上增强 style 约束并扩大 command：`style_floor 0.5 -> 0.1`，速度 `0.25~2.6 -> 0.25~5.0 m/s`，yaw `±0.8 -> ±2.0 rad/s`；用于观察更强 walk/run prior 是否能减少弓背，同时测试高速和大转向能力。 |  |
| 2026-06-23 20:44 | `smp_body_velocity_lafan_walk_posture_g1/2026-06-23_20-44-10_smp_body_velocity_lafan_walk_posture_g1` | `Smp-G1-BodyVelocity-LafanWalk-Posture-v0` | 在 LAFAN Walk strong-style 实验基础上新增显式姿态正则：`base_upright=-0.2`、`root_height=-0.2,target_height=0.74`、`waist_pitch_roll=-0.1`、`action_rate=-0.005`；command 保持速度 `0.15~1.6 m/s`、yaw `±0.8 rad/s`，用于验证弓背问题是否来自任务 reward 缺少上身/腰部姿态约束。 |  |
| 2026-06-23 22:06 | `smp_body_velocity_lafan_walk_g1/2026-06-23_22-06-49_smp_body_velocity_lafan_walk_g1` | `Smp-G1-BodyVelocity-LafanWalk-v0` | 在 LAFAN Walk strong-style 实验基础上去掉 style floor：`style_floor 0.1 -> 0.0`；command 保持速度 `0.15~1.6 m/s`、yaw `±0.8 rad/s`；prior 保持 `lafan_walk_local_norm.pt`；用于测试纯 `task * style` product 是否能进一步减少弓背和异常步态。 |  |
| 2026-06-23 22:09 | `smp_body_velocity_lafan_walk_run_g1/2026-06-23_22-09-00_smp_body_velocity_lafan_walk_run_g1` | `Smp-G1-BodyVelocity-LafanWalkRun-v0` | 在 LAFAN WalkRun strong-style 实验基础上去掉 style floor：`style_floor 0.1 -> 0.0`；command 保持速度 `0.25~5.0 m/s`、yaw `±2.0 rad/s`；prior 保持 `lafan_walk_run_local_nrom.pt`；用于测试 walk/run 纯 `task * style` product 是否能改善弓背，并观察低速 walk 到高速 run 的过渡。 |  |
| 2026-06-24 00:24 | `smp_body_velocity_amp_all_g1/2026-06-24_00-24-31_smp_body_velocity_amp_all_g1` | `Smp-G1-BodyVelocity-AmpAll-v0` | 新建 AMP All prior 的 BodyVelocity 任务；prior 使用 `amp_all.pt`；command 速度 `0.0~5.0 m/s`，yaw `±2.0 rad/s`；`style_floor=0.0`，不使用 soft blending 常数项；用于测试全量 motion prior 在宽速度和大转向范围下是否比单一 walk/walk-run prior 更稳定，并观察是否改善弓背和低速异常步态。 |  |
