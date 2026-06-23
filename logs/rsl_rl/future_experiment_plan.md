# SMP BodyVelocity 后续实验计划

更新时间：2026-06-23

本文档基于当前已完成的 BodyVelocity、LAFAN Walk、FootRegularized、ZeroVelocity 等实验，安排后续实验顺序。建议每次实验启动后，在对应 run 目录内继续写独立 `note.md`，并在 `experiment_summary.md` 中补充结论。

## 当前主要问题

已有实验暴露出几个核心问题：

- BodyVelocity 已能实现较好的全向移动，但高速/奔跑时 yaw rate 跟踪较差。
- 低速时容易出现单侧脚尖着地、左右脚不交替的问题。
- 纯 LAFAN walk prior 如果 command 范围过宽，容易出现站着抖腿或无法形成有效移动。
- walk prior 使用本地归一化和 matched command 后更合理，但还需要验证是否能稳定产生低速步态。
- 新训练的 `lafan_walk_run_local_nrom.pt` prior 覆盖 walk/run 数据，适合验证一个 prior 是否能同时支持两种步态和速度过渡。

## 实验 1：LAFAN WalkRun Prior 基础验证

目的：验证新的 walk/run local-normalized prior 是否能在 BodyVelocity 中同时还原 walk 和 run 两种步态。

任务：

```text
Smp-G1-BodyVelocity-LafanWalkRun-v0
```

日志：

```text
logs/rsl_rl/smp_body_velocity_lafan_walk_run_g1/<RUN>
```

配置：

```text
prior: datasets/pretrain_ckpt/lafan_walk_run_local_nrom.pt
speed_min: 0.25 m/s
speed_max: 2.6 m/s
yaw_rate_min: -0.8 rad/s
yaw_rate_max: 0.8 rad/s
stand_sample_prob: 0.0
style_floor: 0.5
reward: task * (0.5 + 0.5 * style)
```

本实验暂时不加入脚部正则，避免将 prior 效果和 foot reward 效果混在一起。

重点观察：

- `0.25-0.6 m/s` 是否能形成左右脚交替的 walk。
- `0.6-1.2 m/s` 是否能自然过渡到快走/慢跑。
- `1.2-2.6 m/s` 是否能形成 run。
- 是否只学到一种步态，例如全程小碎步或全程跑。
- yaw rate 在 `±0.8 rad/s` 内是否稳定。
- 是否仍然出现单侧脚尖着地和明显左右不对称。

## 实验 2：WalkRun 扩大 Command 范围

前置条件：实验 1 能稳定走/跑，且没有明显崩溃。

目的：测试 walk/run prior 的上限和更强转向能力。

建议新建或修改为 ablation task：

```text
speed_min: 0.15 m/s
speed_max: 3.0 m/s
yaw_rate_min: -1.2 rad/s
yaw_rate_max: 1.2 rad/s
style_floor: 0.5
```

重点观察：

- 低速 `0.15-0.25 m/s` 是否退化为抖腿/拖滑。
- 高速 `2.6-3.0 m/s` 是否仍然自然。
- yaw rate 扩大后转向是否明显改善，还是破坏步态。

## 实验 3：WalkRun + FootRegularized

前置条件：实验 1 或实验 2 可以移动，但低速仍有脚尖着地、单脚支撑过久、不交替等问题。

目的：在 walk/run prior 基础上加入脚部正则，改善低速接触形态。

建议新增任务：

```text
Smp-G1-BodyVelocity-LafanWalkRun-FootRegularized-v0
```

继承实验 1 或实验 2 的 command/prior/reward，再加入：

```text
feet_air_time: 0.25
feet_slide: -0.1
support_foot_tilt: -0.1
persistent_single_support: -0.2
action_rate: -0.005
```

重点观察：

- 低速是否更容易交替。
- 脚底是否更平，脚尖支撑是否减少。
- 是否因为正则过强导致速度跟踪下降。
- 是否出现跳步、踢腿或高频动作。

如果速度跟踪明显下降，优先降低：

```text
persistent_single_support: -0.2 -> -0.1
support_foot_tilt: -0.1 -> -0.05
```

## 实验 4：Style Floor 权重扫描

目的：找到 task tracking 和 style prior 之间更合适的平衡。

建议在 WalkRun 基础任务上做三组：

```text
style_floor = 0.3
style_floor = 0.5
style_floor = 0.7
```

预期：

- `0.3`：更强 style，可能步态更像数据，但速度/yaw 跟踪可能弱。
- `0.5`：当前折中版本。
- `0.7`：更强 task，可能更容易跟踪 command，但风格约束变弱。

重点观察：

- 是否出现站着抖腿刷 reward。
- 不同速度段的 tracking error。
- 步态自然性和速度跟踪的折中点。

## 实验 5：Yaw Rate 专项改进

前置条件：walk/run 主体运动稳定后再做。

目的：针对 BodyVelocity 长期存在的 yaw rate 跟踪不足问题。

可尝试：

```text
yaw_rate_weight: 0.25 -> 0.35 或 0.4
lin_vel_weight: 0.75 -> 0.65 或 0.6
yaw_rate_err_scale: 1.0 -> 1.5
```

建议不要和 prior/foot 正则同时大改。一次只改 yaw 相关项。

重点观察：

- `error_yaw_rate` 是否下降。
- 高速转向是否更好。
- 是否牺牲线速度跟踪或导致上身扭动过大。

## 实验 6：ZeroVelocity / Stand 与 WalkRun 的衔接

前置条件：walk/run 任务稳定后再考虑。

目的：最终需要一个能覆盖站立、走、跑的策略或策略组合。

建议路线：

1. 继续保留 `ZeroVelocity` 作为独立站立 teacher。
2. 使用 WalkRun 作为移动 teacher。
3. 后续通过 dual distillation 合并，而不是直接让 walk/run prior 学 stand。

原因：

- 当前 `g1_walk_run` 数据没有 stand。
- 直接在 WalkRun command 中采样 0 速度，容易造成 prior 和 task 冲突。
- 独立站立 teacher + 移动 teacher 的分布更清晰，便于蒸馏。

## 推荐执行顺序

1. 跑 `Smp-G1-BodyVelocity-LafanWalkRun-v0` 基础实验。
2. 若基础实验能自然 walk/run，做扩大 command ablation。
3. 若低速脚部仍有问题，做 WalkRun + FootRegularized。
4. 在稳定版本上扫描 `style_floor`。
5. 最后单独调 yaw rate reward。
6. 需要站立能力时，用 ZeroVelocity teacher + WalkRun teacher 做蒸馏。

