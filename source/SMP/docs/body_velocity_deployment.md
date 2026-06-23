# G1 BodyVelocity 96-D Policy 部署对接文档

## 1. 适用范围

本文档对应 `Smp-G1-BodyVelocity-v0` 的可部署 Actor：

- Actor observation：`96` 维
- Actor action：`29` 维
- 不输入 `base_lin_vel_b`
- 单帧前馈 MLP，无 observation history
- 控制频率：`50 Hz`，每 `20 ms` 推理一次
- command：机体系 `[v_x, v_y, yaw_rate]`

以实验 `2026-06-21_20-06-35_smp_body_velocity_g1` 为例，checkpoint 已验证为：

```text
normalizer dimension: 96
first actor layer:    [512, 96]
actor output:         29
```

不同实验的 command 范围可能不同，部署前应同时核对对应运行目录中的 `params/env.yaml`。

## 2. 推理接口

ONNX 输入输出：

```text
input name:  obs
input shape: [batch, 96], float32

output name:  actions
output shape: [batch, 29], float32
```

单机器人部署时：

```text
obs shape     = [1, 96]
actions shape = [1, 29]
```

导出的 ONNX 已包含训练得到的 observation mean/std，不要在外部再次归一化。

## 3. 96 维 Observation 布局

所有量必须按下表顺序拼接，不能按实机 SDK 的默认关节顺序直接填充。

| 索引（左闭右开） | 维度 | 内容 | 单位 |
|---|---:|---|---|
| `[0:3]` | 3 | `base_ang_vel_b = [wx, wy, wz]` | rad/s |
| `[3:6]` | 3 | `projected_gravity_b = [gx, gy, gz]` | unit vector |
| `[6:35]` | 29 | `joint_pos - default_joint_pos` | rad |
| `[35:64]` | 29 | `joint_vel - default_joint_vel` | rad/s |
| `[64:93]` | 29 | 上一控制帧的原始 policy action | dimensionless |
| `[93:96]` | 3 | `[v_x_body, v_y_body, yaw_rate]` | m/s, m/s, rad/s |

总维度：

```text
3 + 3 + 29 + 29 + 29 + 3 = 96
```

### 3.1 Base angular velocity

`base_ang_vel_b` 是根节点/骨盆坐标系表达的角速度，不是世界坐标系角速度：

```text
[roll_rate, pitch_rate, yaw_rate]
```

推荐由 IMU gyroscope 获取，并转换到 URDF root frame。单位必须为 `rad/s`。

### 3.2 Projected gravity

训练中的定义为：

```text
projected_gravity_b = R_world_to_body * [0, 0, -1]
```

机器人直立时应接近：

```text
[0, 0, -1]
```

这是归一化重力方向，不要传入 `[0, 0, -9.81]`。四元数顺序、IMU 安装外参和 body/world 旋转方向必须经过静态姿态测试确认。

### 3.3 Joint position

输入 policy 的不是绝对关节角，而是：

```text
joint_pos_rel[i] = measured_joint_pos[i] - default_joint_pos[i]
```

训练时加入了 encoder bias 和均匀观测噪声用于鲁棒化。实机部署时使用真实编码器读数，不要人为添加训练噪声或随机 bias。

### 3.4 Joint velocity

默认关节速度为 0，因此：

```text
joint_vel_rel[i] = measured_joint_vel[i]
```

如实机速度估计噪声较大，应先进行低延迟滤波，但不要引入明显相位延迟。

### 3.5 Last action

`obs[64:93]` 必须是上一帧网络输出的 29 维原始 action：

```text
last_action = previous_policy_output
```

它不是：

- 上一帧绝对关节目标；
- 当前关节位置；
- 经过 action scale 后的关节偏移；
- PD 控制器实际输出的 torque。

启动第一帧建议初始化为全零。

### 3.6 Body-frame command

Command 定义：

```text
[v_x_body, v_y_body, yaw_rate]
```

坐标约定：

- `+x`：向前；
- `+y`：向左；
- `+yaw`：绕 `+z` 逆时针/左转。

当前 2026-06-21 实验的训练范围：

```text
sqrt(v_x^2 + v_y^2): 0.0–3.0 m/s
yaw_rate:            -2.0–2.0 rad/s
```

部署端应限制 command 在对应 checkpoint 的训练范围内，并对遥控指令增加合理的 rate limit。

## 4. 29 维关节顺序、默认姿态和 Action Scale

Observation 中的 joint position、joint velocity、last action，以及 policy 输出 action，全部使用同一顺序：

| i | Joint | Default position | Action scale |
|---:|---|---:|---:|
| 0 | `left_hip_pitch_joint` | -0.312 | 0.5475464652 |
| 1 | `left_hip_roll_joint` | 0.0 | 0.3506614664 |
| 2 | `left_hip_yaw_joint` | 0.0 | 0.5475464652 |
| 3 | `left_knee_joint` | 0.669 | 0.3506614664 |
| 4 | `left_ankle_pitch_joint` | -0.363 | 0.4385773139 |
| 5 | `left_ankle_roll_joint` | 0.0 | 0.4385773139 |
| 6 | `right_hip_pitch_joint` | -0.312 | 0.5475464652 |
| 7 | `right_hip_roll_joint` | 0.0 | 0.3506614664 |
| 8 | `right_hip_yaw_joint` | 0.0 | 0.5475464652 |
| 9 | `right_knee_joint` | 0.669 | 0.3506614664 |
| 10 | `right_ankle_pitch_joint` | -0.363 | 0.4385773139 |
| 11 | `right_ankle_roll_joint` | 0.0 | 0.4385773139 |
| 12 | `waist_yaw_joint` | 0.0 | 0.5475464652 |
| 13 | `waist_roll_joint` | 0.0 | 0.4385773139 |
| 14 | `waist_pitch_joint` | 0.0 | 0.4385773139 |
| 15 | `left_shoulder_pitch_joint` | 0.2 | 0.4385773139 |
| 16 | `left_shoulder_roll_joint` | 0.2 | 0.4385773139 |
| 17 | `left_shoulder_yaw_joint` | 0.0 | 0.4385773139 |
| 18 | `left_elbow_joint` | 0.6 | 0.4385773139 |
| 19 | `left_wrist_roll_joint` | 0.0 | 0.4385773139 |
| 20 | `left_wrist_pitch_joint` | 0.0 | 0.0745008703 |
| 21 | `left_wrist_yaw_joint` | 0.0 | 0.0745008703 |
| 22 | `right_shoulder_pitch_joint` | 0.2 | 0.4385773139 |
| 23 | `right_shoulder_roll_joint` | -0.2 | 0.4385773139 |
| 24 | `right_shoulder_yaw_joint` | 0.0 | 0.4385773139 |
| 25 | `right_elbow_joint` | 0.6 | 0.4385773139 |
| 26 | `right_wrist_roll_joint` | 0.0 | 0.4385773139 |
| 27 | `right_wrist_pitch_joint` | 0.0 | 0.0745008703 |
| 28 | `right_wrist_yaw_joint` | 0.0 | 0.0745008703 |

单位：default position 和 action scale 均以 `rad` 为关节目标单位。

## 5. Action 后处理

Policy 输出是无量纲的相对关节动作。绝对关节位置目标：

```text
q_target[i] = default_joint_pos[i] + action_scale[i] * action[i]
```

训练配置中：

```text
use_default_offset = true
offset = 0
action clip = null
```

因此不能把网络输出直接当作绝对关节角发送。

仿真中的 `BiasedJointPositionAction` 会对隐藏的 encoder bias 做补偿，以模拟实机带偏置编码器闭环。实机侧正常发送上述 `q_target` 即可，不要再手工减去训练时的随机 bias。

## 6. PD 控制与时序

高层 policy 频率：

```text
simulation dt = 0.005 s
decimation    = 4
policy dt     = 0.020 s
policy rate   = 50 Hz
```

每次 policy 推理得到新的 `q_target` 后，在接下来的 20 ms 内保持该目标，由底层关节 PD 高频执行。

仿真 PD 参数定义在 `source/SMP/SMP/robots/g1.py`。部署端应尽量匹配对应关节组的 stiffness、damping、effort limit 和 velocity limit；若使用 Unitree SDK 的不同控制增益，需要重新验证 sim-to-real 稳定性。

## 7. ONNX 导出

示例：

```bash
cd /home/cyq/SMP_isaaclab

python scripts/rsl_rl/export_onnx.py \
  --checkpoint logs/rsl_rl/smp_body_velocity_g1/<RUN>/model_9999.pt \
  --output-dir logs/rsl_rl/smp_body_velocity_g1/<RUN>/exported \
  --output-name policy.onnx
```

导出器会：

1. 读取 `actor_state_dict`；
2. 嵌入 observation mean/std；
3. 导出 deterministic actor mean；
4. 自动从 checkpoint 推断输入和输出维度。

导出时应看到：

```text
Actor input dim: 96, action dim: 29
```

## 8. 每帧部署伪代码

```python
# 1. Read sensors and map joints to the canonical 29-joint order.
ang_vel_b = imu_angular_velocity_in_root_frame()       # [3]
gravity_b = normalized_gravity_in_root_frame()         # [3]
q = measured_joint_positions_in_policy_order()         # [29]
dq = measured_joint_velocities_in_policy_order()       # [29]

# 2. Construct the raw 96-D observation.
obs = concatenate([
    ang_vel_b,
    gravity_b,
    q - default_joint_pos,
    dq,
    previous_raw_action,
    command_body,
]).astype(float32)

assert obs.shape == (96,)

# 3. ONNX contains observation normalization.
raw_action = onnx_policy(obs[None, :])[0]              # [29]

# 4. Convert policy action to joint-position targets.
q_target = default_joint_pos + action_scale * raw_action

# 5. Apply safety limits, send q_target, and retain the raw action.
q_target = safety_filter(q_target)
send_joint_position_targets(q_target)
previous_raw_action = raw_action
```

## 9. 启动与安全建议

1. 机器人先进入 default pose，再启用 policy。
2. 启动时 command 设为 `[0, 0, 0]`，`last_action` 设为全零。
3. 在 1–2 秒内从当前关节位置平滑过渡到 policy target，禁止直接跳变。
4. 实机首次测试应悬吊或使用保护架，并限制 command 和 action rate。
5. 对关节位置、速度、力矩、机体倾角、通信超时设置独立硬保护。
6. 检测到 NaN、推理超时或 IMU 异常时，立即退出 policy 控制。
7. 训练中没有 action clip；实机必须使用关节软限位与安全滤波，但应记录限幅触发率以评估部署分布偏移。

## 10. 对接自检清单

- [ ] ONNX 显示 `96 -> 29`。
- [ ] 关节顺序与本文 29 关节表完全一致。
- [ ] 静止直立时 projected gravity 接近 `[0, 0, -1]`。
- [ ] angular velocity 使用 root/body frame，单位为 rad/s。
- [ ] joint position 输入已减 default pose。
- [ ] last action 使用上一帧原始网络输出。
- [ ] command 使用 body frame，单位和符号一致。
- [ ] ONNX 外部没有重复做 observation normalization。
- [ ] 网络输出经过 action scale 和 default offset 后才成为关节目标。
- [ ] 高层推理稳定运行在 50 Hz。
- [ ] 已配置关节、姿态、通信和急停保护。

