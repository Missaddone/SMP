# SMP Migration Status

This document records the current SMP migration state from `D:\OpenSource_Project\smp-master` into this IsaacLab extension. It is intended as handoff context for running and debugging the project on another machine with IsaacLab/Isaac Sim available.

## Scope

The migration follows the `smp-master` G1 implementation, not the original MimicKit implementation.

Do not migrate MimicKit-specific components unless the project direction changes:

- `TinyMDMModel`
- `diffusers` schedulers
- MimicKit `MotionLib`
- `compute_disc_obs`
- MimicKit `SMPAgent`
- MimicKit `.pkl` motion dataset pipeline

The current route is:

```text
G1 retargeted CSV
  -> SMP 59-dim NPZ windows
  -> norm stats
  -> diffusion prior pretrain
  -> load frozen prior in IsaacLab env
  -> SMP reward / GSI / downstream tasks
```

## Migrated Components

### Base IsaacLab Environment

Updated:

```text
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/smp_catchball_env_cfg.py
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/__init__.py
source/SMP_catchball/SMP_catchball/robots/g1.py
source/SMP_catchball/SMP_catchball/robots/smpl.py
```

Changes:

- Replaced the original cartpole template robot with `G1_CYLINDER_CFG`.
- Fixed robot asset imports from `whole_body_tracking.assets` to `SMP_catchball.assets`.
- Kept `SmpEnvCfg` as the reusable base SMP environment.
- Registered the base template task and added `Smp-G1-Forward-v0`.
- Added GSI startup/reset/refresh events to the base config.
- Added `SmpG1ForwardEnvCfg` as the first downstream task.

### SMP Core Model / Pretrain

Migrated or added:

```text
source/SMP_catchball/SMP_catchball/smp/model.py
source/SMP_catchball/SMP_catchball/smp/scheduler.py
source/SMP_catchball/SMP_catchball/smp/dataset.py
source/SMP_catchball/SMP_catchball/smp/pretrain.py
source/SMP_catchball/SMP_catchball/smp/pretrain_cfg.py
source/SMP_catchball/SMP_catchball/smp/utils.py
source/SMP_catchball/SMP_catchball/smp/feature_to_state.py
```

Capabilities:

- Load and train `DiffusionDenoiser`.
- Use custom cosine `DDPMScheduler`.
- Read NPZ files with `windows` shaped `(N, W, F)`.
- Normalize features using `q_low/q_high`.
- Save checkpoints compatible with `load_denoiser`.
- Load original `smp-master` checkpoints such as `pretrained_loco.pt`.
- Convert SMP 6D rotation features back to pelvis/joint/EE trajectories.

### Pretrain Scripts

Added:

```text
scripts/smp_csv_to_npz.py
scripts/smp_compute_norm_stats.py
scripts/smp_pretrain.py
scripts/smp_sample_prior.py
```

Purpose:

- `smp_csv_to_npz.py`: IsaacLab version of `smp-master/scripts/csv_to_npz.py`.
- `smp_compute_norm_stats.py`: computes `q_low/q_high` from NPZ windows.
- `smp_pretrain.py`: trains diffusion prior from NPZ windows.
- `smp_sample_prior.py`: offline prior sampling, saves sampled denormalized windows to `outputs/`.

Expected data flow:

```powershell
python scripts\smp_csv_to_npz.py --headless `
  --input-dir datasets\g1 `
  --output-dir datasets\g1_npz `
  --input-fps 30 `
  --output-fps 50

python scripts\smp_compute_norm_stats.py `
  --input-dir datasets\g1_npz `
  --output datasets\g1_norm_stats.npz

python scripts\smp_pretrain.py `
  --data-dir datasets\g1_npz `
  --norm-stats-file datasets\g1_norm_stats.npz `
  --name g1_lafan
```

### SMP Reward / GSI / Commands

Updated or added:

```text
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/mdp/events.py
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/mdp/rewards.py
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/mdp/commands.py
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/mdp/terminations.py
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/mdp/__init__.py
```

Capabilities:

- `init_smp_state`: load frozen prior, allocate feature buffer, create GSI sample pool.
- `_ddpm_sample`: generate denormalized motion windows from prior.
- `gsi_reset`: reset envs using sampled prior states.
- `gsi_refresh`: periodically refresh GSI sample pool.
- `smp_guidance_reward`: compute SDS-style SMP reward from online motion feature window.
- `task_smp_product`: multiplicative task reward gated by SMP reward.
- `SteeringCommand`: target direction/speed/facing command for forward/steering tasks.
- `steering_target_velocity` and `steering_face_direction`.
- `root_height_below_minimum` termination.

## Current Dataset

The local dataset is:

```text
datasets/g1/*.csv
```

Observed format:

- 40 CSV files.
- No header.
- Each row has 36 float columns:

```text
root_pos(3), root_quat(4), joint_pos(29)
```

- Total frames observed: 264,705.
- CSV root quaternion appears to be `xyzw`; `smp_csv_to_npz.py` defaults to `--quat-order xyzw` and converts to `wxyz`.

The output NPZ must match SMP G1 feature layout:

```text
root_pos(3)
root_rot(6)
joint_pos(29)
ee_pos(15)
root_lin_vel(3)
root_ang_vel(3)
```

Total feature dimension:

```text
3 + 6 + 29 + 15 + 3 + 3 = 59
```

Tracked end-effector body names:

```text
left_ankle_roll_link
right_ankle_roll_link
torso_link
left_wrist_yaw_link
right_wrist_yaw_link
```

## What Has Not Been Migrated

### Full Runtime Validation

This machine could not run IsaacLab/Isaac Sim, so the following are unverified:

- `scripts/smp_csv_to_npz.py` launching IsaacLab and generating real NPZs.
- `Smp-G1-Forward-v0` env creation.
- GSI writing root/joint state at reset.
- `smp_guidance_reward` running during env step.
- `SteeringCommand` compatibility with the installed IsaacLab command manager.
- Articulation data field names:
  - `root_link_pos_w` vs `root_pos_w`
  - `root_link_quat_w` vs `root_quat_w`
  - `body_link_pos_w` vs `body_pos_w`

### Remaining Downstream Tasks

Only forward locomotion is scaffolded:

```text
Smp-G1-Forward-v0
```

Still missing:

```text
Smp-G1-Steering-v0
Smp-G1-Location-v0
Smp-G1-Getup-v0
Catchball-specific task
```

Steering should be the easiest next task because `SteeringCommand` and steering rewards are already present.

Location still needs:

- goal command
- goal observation
- location reward
- goal resampling logic

Getup still needs:

- getup-specific rewards
- getup-specific termination/reset details
- appropriate getup prior checkpoint

### Domain Randomization / Robustness Events

Not migrated from `smp-master`:

- push disturbance
- foot friction randomization
- encoder bias
- base COM randomization
- self-collision/contact termination

These should be implemented using IsaacLab-native events and sensors.

### Full Visualizer

Original `smp-master/scripts/generate_viz.py` used `mjlab` + `viser`.

Current replacement is only offline:

```text
scripts/smp_sample_prior.py
```

An IsaacLab viewer version is still missing.

## Recommended First Runtime Test Plan

Run these on the machine with IsaacLab/Isaac Sim available.

### 1. Confirm Task Registration

```powershell
python scripts\list_envs.py
```

Expected:

```text
Smp-G1-Forward-v0
Template-Smp-Catchball-v0
```

If registration fails, inspect:

```text
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/__init__.py
source/SMP_catchball/SMP_catchball/__init__.py
```

### 2. Convert CSV To NPZ

```powershell
python scripts\smp_csv_to_npz.py --headless `
  --input-dir datasets\g1 `
  --output-dir datasets\g1_npz `
  --input-fps 30 `
  --output-fps 50
```

Expected output:

```text
datasets/g1_npz/*.npz
```

Each NPZ should contain:

```python
windows.shape == (N, 10, 59)
```

Likely issues:

- IsaacLab scene initialization API mismatch.
- `Articulation.write_root_state_to_sim` signature mismatch.
- `body_link_pos_w` field name mismatch.
- G1 body/joint names not found.

### 3. Compute Norm Stats

```powershell
python scripts\smp_compute_norm_stats.py `
  --input-dir datasets\g1_npz `
  --output datasets\g1_norm_stats.npz
```

Expected:

```text
datasets/g1_norm_stats.npz
q_low.shape == (59,)
q_high.shape == (59,)
```

### 4. Smoke Pretrain

Run a short CPU/GPU smoke test first:

```powershell
python scripts\smp_pretrain.py `
  --data-dir datasets\g1_npz `
  --norm-stats-file datasets\g1_norm_stats.npz `
  --name smoke `
  --num-epochs 1 `
  --batch-size 64 `
  --num-noise-samples 1
```

Expected:

```text
logs/pretrain/smoke/<timestamp>/pretrained.pt
```

### 5. Sample Prior Offline

```powershell
python scripts\smp_sample_prior.py `
  --ckpt-path logs\pretrain\smoke\<timestamp>\pretrained.pt `
  --output outputs\smoke_prior_samples.npz
```

Expected:

```python
windows.shape == (16, 10, 59)
```

### 6. Test Env Creation

Use the original `smp-master` prior first or a newly trained smoke checkpoint.

Current default checkpoint path in `SmpEnvCfg` resolves to:

```text
<project-root>/datasets/pretrain_ckpt/pretrained_loco.pt
```

If that path does not exist on the runtime machine, change:

```text
source/SMP_catchball/SMP_catchball/tasks/manager_based/smp_catchball/smp_catchball_env_cfg.py
```

Then run:

```powershell
python scripts\random_agent.py --headless --task Smp-G1-Forward-v0 --num_envs 4
```

Likely first runtime fixes:

- Command manager config API.
- GSI reset event order.
- Tensor device mismatch.
- IsaacLab articulation data field names.

### 7. Disable GSI If Needed

If GSI blocks first env creation, temporarily disable these in `EventCfg`:

```python
gsi_reset
gsi_refresh
```

And set:

```python
"gsi_buffer_size": 0
```

Then test `smp_guidance_reward` alone first.

## Recommended Next Implementation Order

1. Runtime-fix `smp_csv_to_npz.py`.
2. Generate real `datasets/g1_npz`.
3. Run norm stats and smoke pretrain.
4. Runtime-fix `Smp-G1-Forward-v0` without GSI if necessary.
5. Re-enable and fix GSI.
6. Add `Smp-G1-Steering-v0`.
7. Add `Smp-G1-Location-v0`.
8. Add `Smp-G1-Getup-v0`.
9. Build catchball-specific task on top of `SmpEnvCfg`.

## Notes For Future Codex Sessions

- Prefer `smp-master` as the reference, not MimicKit.
- Preserve the 59-dim G1 feature layout unless intentionally retraining all downstream components.
- Keep quaternion order explicit:
  - CSV appears to be `xyzw`.
  - IsaacLab/SMP code uses `wxyz`.
- Do not commit datasets, outputs, logs, or temporary smoke-test files.
- Use `outputs/` for generated samples; it is already ignored.
- If a runtime issue appears in IsaacLab APIs, patch the local wrapper functions first:
  - `_data_attr` in rewards/scripts
  - `_heading_w` in commands/rewards
  - `_prime_sim_and_buffer` in events
