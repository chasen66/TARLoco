# TARLoco 训练日志

记录每次正式训练的关键信息，便于复盘与对比。

---

## 记录模板

每条 Run 固定包含以下六个部分：

```
### 基本信息     ← 任务/模型/起点/超参/文件
### 上次改动     ← 相比上一 Run 改了什么
### 改动原因     ← 为什么这样改
### 训练结果     ← 指标、曲线观察
### 发现的问题   ← 本次训练暴露的 bug / 异常 / 不足
### 状态         ← 完成 / 中止 / 进行中
```

---

## Run 01 — `2026-03-26_21-36-05`

### 基本信息

| 项目 | 内容 |
|------|------|
| **任务** | `go1-train-teacher-rough` |
| **模型** | Teacher MLP |
| **起点** | 从头训练（scratch） |
| **num_envs** | 4096（默认） |
| **max_iterations** | 20000 |
| **num_steps_per_env** | 24 |
| **seed** | 0 |
| **wandb note** | （空） |
| **日志文件** | `logs/nohup/teacher_s0.log` |

### 上次改动

首次训练，无对比基准。

### 改动原因

首次训练，无。

### 训练结果

- 训练完成（model_19999.pt）
- terrain_level、reward 未重点监控

### 发现的问题

- **[Bug] `flatten_history_dim = False`**：history_length=1 时 obs shape 为 (N,1,D)，actor 输出 (N,1,12) 而非 (N,12)，action dim 与环境期望不符。训练期间未报错（因为没有触发 action shape 检查），事后排查才发现。此 checkpoint **不可用于部署**。
- **`flat_orientation_l2 = 0.0`**：姿态惩罚权重为 0，机器人从未被引导保持躯干水平，对粗糙地形适应能力有限。

### 状态

完成（但 checkpoint 存在已知 bug，不可用）

---

## Run 02 — `2026-03-27_16-41-40`

### 基本信息

| 项目 | 内容 |
|------|------|
| **任务** | `go1-train-teacher-rough` |
| **模型** | Teacher MLP |
| **起点** | 从头训练（scratch） |
| **num_envs** | 8192 |
| **max_iterations** | 20000 |
| **num_steps_per_env** | 24 |
| **seed** | 0 |
| **wandb note** | `Teacher_S0_8kenv` |
| **wandb run name** | `Teacher_S0_8kenv_9` |
| **日志文件** | `logs/nohup/teacher_s0_8kenv.log` |

### 上次改动

- `flatten_history_dim`: False → **True**（`env_cfg.py`，policy + critic 均改）
- `num_envs`: 4096 → **8192**
- 新增必要环境变量：`PYTHONPATH=/home/chasen/TARLoco`、`WANDB_ENTITY=achasen9981-zhejiang-university-of-technology`

### 改动原因

- `flatten_history_dim`：修复 Run 01 发现的 action shape bug，Teacher history_length=1 时必须将 obs 展平才能得到正确 action 维度。
- `num_envs 8192`：GPU（RTX 5070 Ti 16GB）在 4096 envs 时只用了 8.5GB / 16GB（69% 利用率），提升到 8192 后约 11GB，吞吐量翻倍，有效加速收敛。
- 环境变量：修复 PYTHONPATH 未设置导致 `ModuleNotFoundError`，以及 wandb 因 entity 指向原作者账号而返回 404 的问题。

### 训练结果

- 训练完成，耗时约 25 小时
- `mean_reward` ≈ 18–20（论文预期 21–22，差距约 10%）
- `terrain_level` ≈ 2.5，plateau 后未能突破 3（论文预期 5+）
- `base_contact` 从 80 降至 ~2，步态基本稳定
- `track_lin_vel_xy` ≈ 1.1/1.2，线速度跟踪良好
- `error_vel_yaw` ≈ 0.45，偏航控制误差较高
- Checkpoint 可用：`model_19999.pt`、`model_best.pt`

### 发现的问题

- **`flat_orientation_l2` 全程为 0**：`base.py` 中该奖励项权重为 0.0（标注为"optional penalty"），Teacher 配置未覆盖。机器人从未接收姿态惩罚，在高难度地形上无法稳定躯干，导致 curriculum 无法推进到 level 3+。这是 terrain_level 卡在 2.5 的根本原因。
- **step ~11k 出现 surrogate loss 尖峰**：策略短暂不稳定，触发 NaN 回退逻辑加载 best model，此后 terrain_level 出现回落。
- **learning_rate 高度振荡**：adaptive schedule 下 KL 散度频繁超标，lr 在 0.0002–0.001 之间震荡，说明策略更新步长不够稳定。

### 状态

完成

---

## Run 03 — `2026-03-30_10-40-24`（中止）

### 基本信息

| 项目 | 内容 |
|------|------|
| **任务** | `go1-train-teacher-rough` |
| **模型** | Teacher MLP |
| **起点** | Resume from Run 02 `model_19999.pt` |
| **num_envs** | 8192 |
| **max_iterations** | 10000（额外补训） |
| **num_steps_per_env** | 48 |
| **wandb note** | `Teacher_MLP` |
| **日志文件** | `logs/nohup/teacher_resume.log` |

### 上次改动

- `flat_orientation_l2` weight: 0.0 → **-1.0**（`base.py`，影响所有任务）
- `num_steps_per_env`: 24 → **48**
- 训练方式：全新 → **Resume**（继承 Run 02 最终权重）
- `max_iterations` 由 20000 降为 **10000**（补训）

### 改动原因

- `flat_orientation_l2 = -1.0`：修复 Run 02 发现的姿态惩罚缺失问题，引导机器人在粗糙地形保持躯干水平，预期推进 terrain_level。
- `num_steps_per_env 48`：期望更长 rollout 带来更好的时序信用分配，加速地形课程收敛，从而以更少 iterations 完成训练。
- Resume：基础步态已习得，无需从头训练，节约时间。
- `max_iterations 10000`：与 num_steps_per_env 翻倍配合，期望总时间减半。

### 训练结果

未获得有效结果，启动后立即中止。

### 发现的问题

- **num_steps_per_env 翻倍抵消了迭代次数减半的收益**：每次迭代 collection time 从 ~4.5s 增至 ~9s，10000 iter × 9s = 25h，与全新训练相同，未达到缩短训练时间的目的。

### 状态

中止（配置失误，未产生有效 checkpoint）

---

## Run 04 — `2026-03-30_10-45-37`（进行中）

### 基本信息

| 项目 | 内容 |
|------|------|
| **任务** | `go1-train-teacher-rough` |
| **模型** | Teacher MLP |
| **起点** | Resume from Run 02 `model_19999.pt` |
| **num_envs** | 8192 |
| **max_iterations** | 8000（额外补训，总迭代数至 27999） |
| **num_steps_per_env** | 24 |
| **seed** | 0 |
| **wandb note** | `Teacher_MLP` |
| **wandb run name** | `2026-03-30_10-45-37_Teacher_MLP` |
| **日志文件** | `logs/nohup/teacher_resume.log` |

### 上次改动

- `num_steps_per_env`: 48 → **24**（回退至原始值）
- `max_iterations`: 10000 → **8000**

### 改动原因

- `num_steps_per_env 24`：修正 Run 03 的失误。保持原始 collection time（~4.5s/iter），8000 iter × 4.5s ≈ 10h，实现约 60% 的时间节省。
- `max_iterations 8000`：基础步态已习得，orientation 惩罚加入后预计能在 8000 步内推动 terrain_level 从 2.5 升至 4+，无需再跑 20000 步。

### 训练结果

训练中，等待完成后更新。

**前 20 步观察**（快速收敛信号）：
- iter 1: reward -0.79（orientation 惩罚刚加入，短暂下降）
- iter 8: reward 1.96
- iter 16: reward 6.77（步态快速适应新奖励结构）

### 发现的问题

待训练完成后补充。

### 状态

进行中（预计 ~10 小时）

---

## 待训练

| Run | 模型 | 任务名 | 起点 | 关键命令参数 |
|-----|------|--------|------|-------------|
| Run 05 | TAR-RNN | `go1-train-tar-rnn-rough` | scratch | `--num_envs 8192 --max_iterations 20000 --note TAR_RNN` |
| Run 06 | HIMLoco | `go1-train-him-rough` | scratch | `--num_envs 8192 --max_iterations 20000 --note HIMLoco` |

---

## 代码变更摘要

| 文件 | 变更内容 | 改动原因 | 生效自 |
|------|---------|---------|--------|
| `exts/tarloco/tasks/envs/env_cfg.py` | `flatten_history_dim` False → True（Teacher policy+critic） | 修复 action shape bug | Run 02 |
| `exts/tarloco/tasks/algorithms/ppo_cfg.py` | `wandb_entity` 改为本地账号 | 修复 wandb 404 错误 | Run 02 |
| `exts/tarloco/utils/logger.py` | wandb run name 格式改为 `{时间戳}_{note}` | 便于在 wandb 中直接识别模型与时间 | Run 04 |
| `exts/tarloco/tasks/envs/base.py` | `flat_orientation_l2` weight 0.0 → -1.0 | 修复地形难度无法提升的根本原因 | Run 04 |
