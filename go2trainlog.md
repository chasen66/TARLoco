# TARLoco Go2 训练日志

记录 Go2 版本从 Go1 迁移后的关键改动、已修复 bug、当前训练进展与后续计划。

---

## 当前运行（已中止）

### Run 01 — `2026-04-02_13-33-50`

### 基本信息

| 项目 | 内容 |
|------|------|
| 任务 | `go2-train-tar-rnn-rough` |
| 模型 | TAR-RNN (`ActorCriticTarRnn`) |
| 起点 | 从头训练（scratch） |
| num_envs | 4096 |
| max_iterations | 20000 |
| num_steps_per_env | 24 |
| seed | 0 |
| device | `cuda:0` |
| logger | wandb |
| group | `Go2_Tar` |
| nohup 日志 | `logs/nohup/go2_tar_rnn_s0.log` |
| 运行目录 | `logs/rsl_rl/TAR_workspace/2026-04-02_13-33-50` |

### 启动命令

```bash
nohup env PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/train.py \
	--task go2-train-tar-rnn-rough \
	--num_envs 4096 \
	--headless \
	--device cuda:0 \
	--max_iterations 20000 \
	--logger wandb \
	--group Go2_Tar \
	> logs/nohup/go2_tar_rnn_s0.log 2>&1 &
```

### 当前观察（中止前）

- 训练已稳定进入迭代循环（collect/learn 正常打印）
- 未见报错栈、未见 NaN 崩溃信号
- 初始阶段 `rew` 为负属于常见现象（策略尚未收敛）
- 主要 warning 为 Isaac/Omniverse 常见非阻塞告警
- 中止前进度约为 iteration 137，timesteps 约 13.468M

### 状态

已手动停止（用户请求中止并重启为固定 seed 方案）。

---

## 新训练命令（固定全局 seed + terrain seed）

```bash
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/train.py \
	--task go2-train-tar-rnn-rough \
	--num_envs 4096 \
	--headless \
	--device cuda:0 \
	--max_iterations 20000 \
	--seed 0 \
	--logger wandb \
	--group Go2_Tar \
	"env.scene.terrain.terrain_generator.seed=0" \
	> logs/nohup/go2_tar_rnn_s0_seeded.log 2>&1 &
```


PYTHONPATH=/home/chasen/TARLoco python evaluate.py --task go2-eval-tar-rnn-rough --num_envs 200 --num_episodes 10 --load_run 你的run目录 --checkpoint model_best.pt --device cuda:0 --seed 0 --cam_view free "env.commands.base_velocity.ranges.lin_vel_x=[1.0,1.0]" "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"

说明：

- `--seed 0` 固定算法/环境全局随机种子
- `env.scene.terrain.terrain_generator.seed=0` 固定地形生成随机种子

---

## 新评估命令（ID / OOD，固定 seed）

先确认训练目录名（用于 `--load_run`）：

```bash
ls -1 logs/rsl_rl/TAR_workspace | tail -n 5
```

### ID 评估

```bash
PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
	--task go2-eval-tar-rnn-rough \
	--num_envs 200 \
	--num_episodes 10 \
	--load_run <替换为本次训练目录名> \
	--checkpoint model_best.pt \
	--device cuda:0 \
	--seed 0 \
	--cam_view free \
	"env.scene.terrain.terrain_generator.seed=0" \
	"env.commands.base_velocity.ranges.lin_vel_x=[1.0,1.0]" \
	"env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
	"env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"
```

### OOD 评估

```bash
PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
	--task go2-eval-tar-rnn-rough \
	--num_envs 200 \
	--num_episodes 10 \
	--load_run <替换为本次训练目录名> \
	--checkpoint model_best.pt \
	--device cuda:0 \
	--seed 0 \
	--cam_view free \
	"env.scene.terrain.terrain_generator.seed=0" \
	"env.events.add_base_mass.params.mass_distribution_params=[-5.0,15.0]" \
	"env.events.physics_material.params.static_friction_range=[0.05,4.0]" \
	"env.commands.base_velocity.ranges.lin_vel_x=[0.0,2.0]" \
	"env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
	"env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"
```

---

## Go1 -> Go2 迁移改动（代码级）

以下内容为本次 Go2 训练能跑通的关键迁移与适配。

### 1) 任务注册与命名全量切换

- `exts/tarloco/tasks/__init__.py`
	- 将所有 `go1-train-* / go1-eval-*` 任务注册映射替换为 `go2-train-* / go2-eval-*`
	- 覆盖主方法、baseline、teacher、全部 ablation 任务

### 2) Go2 环境类重命名与入口替换

- `exts/tarloco/tasks/envs/env_cfg.py`
	- 环境类从 `*Go1*` 重命名为 `*Go2*`
	- 评估类与训练类同步替换（Tar/Slr/Him/Teacher 及其变体）

### 3) Go2 Runner 配置替换

- `exts/tarloco/tasks/agents/rsl_rl_cfg.py`
	- runner 类从 `Go1...RunnerCfg` 全量替换为 `Go2...RunnerCfg`
	- TAR/SLR/HIM/Teacher 及 no-priv/no-vel 变体全量迁移

### 4) 机器人资产与机体命名适配

- `exts/tarloco/tasks/envs/base.py`
	- 机器人资产：`UNITREE_GO1_CFG` -> `UNITREE_GO2_CFG`
	- 机体 body 名称：`trunk` -> `base`
	- 涉及质量随机化、外力扰动、push 事件、终止条件、height scanner 挂载路径

### 5) 足端接触索引从硬编码改为动态解析

- `exts/tarloco/envs/mdp/observations.py`
	- `feet_contact_z` 从固定索引 `[4,8,12,16]` 改为 `sensor_cfg.body_ids`
- `exts/tarloco/envs/wrappers/evaluate_wrapper.py`
	- 运行时解析 `*_foot` 的 body id，避免 Go1/Go2 链接顺序差异导致评估错误

### 6) TAR 辅助损失与环境数耦合修复

- `exts/tarloco/learning/algorithms/tar_ppo.py`
	- `self.num_envs`：`4096` -> `None`（取消硬编码）
	- `sum(1)` -> `sum(-1)`，兼容不同 batch 维度布局
- `exts/tarloco/learning/runners/on_policy_runner.py`
	- runner 初始化时将 `self.alg.num_envs = self.env.num_envs`

### 7) 脚本层健壮性与 Go2 默认任务更新

- `standalone/tarloco/train.py`
	- 任务读取由 `registry[args_cli.task]` 改为 `registry.get(...)`，不存在任务时给出可读报错
- `standalone/tarloco/evaluate.py`
	- 同步使用 `registry.get(...)`，避免 KeyError
- `scripts/train_seeds.sh`
	- 默认任务改为 `go2-train-tar-rnn-rough`
- `scripts/evaluate_batch.py`
	- 任务键切换到 `go2-eval-*`

---

## 已修复 Bug 汇总

### Bug #1：Recurrent 评估时隐状态未传递（已修复）

- 影响：TAR-RNN 等循环策略评估性能严重失真
- 修复：
	- `exts/tarloco/learning/runners/on_policy_runner.py` 新增 `get_inference_policy_recurrent()`
	- `standalone/tarloco/evaluate.py` 在 step 后根据 `terminated | truncated` 调用 `policy.reset(dones)`

### Bug #2：EvaluationConfigMixin 终止配置拼写错误（已修复）

- 根因：`self.termiantions` 拼写错误导致 eval 终止配置未生效
- 修复文件：`exts/tarloco/tasks/envs/base.py`
- 修复后：使用 `self.terminations = TerminationsEvalCfg()`

### Bug #3：Go2 足端接触力读取硬编码索引不可靠（已修复）

- 根因：硬编码索引仅在特定机器人 link 顺序下成立
- 修复文件：
	- `exts/tarloco/envs/mdp/observations.py`
	- `exts/tarloco/envs/wrappers/evaluate_wrapper.py`
- 修复方式：统一改为传感器 body id 动态解析

### Bug #4：TAR 算法内部 num_envs 硬编码 4096（已修复）

- 影响：切换环境规模时可能导致辅助损失采样逻辑不一致
- 修复文件：
	- `exts/tarloco/learning/algorithms/tar_ppo.py`
	- `exts/tarloco/learning/runners/on_policy_runner.py`

### Bug #5：任务不存在时脚本报错不友好（已修复）

- 修复文件：
	- `standalone/tarloco/train.py`
	- `standalone/tarloco/evaluate.py`
- 修复方式：统一使用 `registry.get(...)` 并输出可用任务列表

### Bug #6：Wandb continue-run 配置读取易抛异常（已修复）

- 修复文件：
	- `exts/tarloco/learning/runners/on_policy_runner.py`
	- `exts/tarloco/learning/runners/him_on_policy_runner.py`
- 修复方式：`self.cfg["wandb_continue_run"]` 改为 `self.cfg.get("wandb_continue_run", False)`

### Bug #7：TAR triplet loss 在 RNN 模式下对错误维度求和（已修复）

- 根因：`pos_loss` 和 `neg_loss` 使用 `.sum(1)`，MLP 模式 tensor 为 `(batch, latent_dim)` 无影响，但 RNN 模式 tensor 为 `(seq_len, batch, latent_dim)` 时 `dim=1` 对 batch 维度求和，triplet loss 语义错误
- 修复文件：`exts/tarloco/learning/algorithms/tar_ppo.py`
- 修复方式：`.sum(1)` → `.sum(-1)`，两种模式下均正确对 `latent_dim` 求和

### Bug #8：`feet_contact_z` 观测项未通过 params 传入 sensor_cfg 导致 body_ids 未解析（已修复）

- 根因：`ObsTerm(func=mdp.feet_contact_z)` 未通过 `params` 传入 `sensor_cfg`，IsaacLab 的 ObservationManager 不会自动 resolve 默认参数中的 `body_names` 正则，导致 `sensor_cfg.body_ids` 为全部 body（19 个），观测维度从预期 `(4,)` 变为 `(19,)`
- 影响：训练输入维度错误，teacher 训练结果无效
- 修复文件：`exts/tarloco/tasks/envs/base.py`
- 修复方式：添加 `params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")}`

---

## 论文 vs 代码对照核查（2026-04-02）

### 核心算法对照 — 全部一致 ✅

| 论文规格 | 代码 | 状态 |
|---------|------|------|
| PPO + GAE (γ=0.99, λ=0.95) | `rsl_rl_cfg.py` | ✅ |
| Adaptive LR [5e-5, 1e-3] | `rsl_rl_cfg.py` | ✅ |
| Latent dim=45, LSTM [256], Dynamics [64] | `rsl_rl_cfg.py` | ✅ |
| Actor/Critic [512,256,128], ELU | `rsl_rl_cfg.py` | ✅ |
| Triplet loss coef=1.0, KL=0.01, clip=0.2 | `rsl_rl_cfg.py` | ✅ |
| 双编码器 (student LSTM + critic MLP) | `ac_tar.py` | ✅ |
| Vel estimator 4 步历史 + sg[·] 梯度阻断 | `ac_tar.py` | ✅ |
| 负样本排除同环境 agent | `tar_ppo.py` | ✅ |
| Critic 同时被 value loss + triplet loss 更新 | `ppo.py` | ✅ |

### Reward 函数对照 — 全部一致 ✅

所有 9 项 reward term 权重与论文 Table II 完全一致。代码额外有 `flat_orientation_l2=-1.0`（论文未列，不影响算法正确性）。

### Domain Randomization 参数偏差

| 参数 | 论文 Table III | 代码 | 状态 |
|-----|--------------|------|------|
| Friction | [0.1, 3.0] | static [0.15, 3.16], dynamic [0.1, 3.0] | ✅ 基本一致 |
| Restitution | [0.0, 1.0] | [0.0, 1.0] | ✅ |
| Payload | [-2, 10] kg | [-2.0, 10.0] | ✅ |
| Ext. Force | ±20 N | **±10 N** | ⚠️ 偏小 |
| Ext. Torque | ±5 N·m | ±5 | ✅ |
| Joint Init Pos | [0.5, 1.5]* | **(1.0, 1.0)** | ⚠️ 无随机化 |

**注**：外力范围和关节初始化随机化与论文有偏差，可能影响 OOD 泛化性能的定量复现，但不影响算法正确性。

### TAR 训练不依赖 Teacher checkpoint

经代码审计确认：TAR 是端到端自监督训练。"Teacher" 信号来自观测空间的不对称性（actor encoder 只看本体感受，critic encoder 能看特权信息），triplet loss 让 actor encoder 的预测逼近 critic encoder 的表征。三个训练任务（Teacher、HIM、TAR RNN）完全独立，可并行。

---

## 已知限制/待处理项

### 部署链路

- **JIT 导出器不适配 TAR 架构**：`exporter.py` 假设模型结构为 `memory_a.rnn + actor`，未处理 TAR 的 `encoder → vel_estimator → actor` 推理链路。部署到 MuJoCo/实机前需编写 TAR 专用导出器。
- **MuJoCo 部署需准备 Go2 MJCF/XML**：IsaacSim 用 USD，需从 Unitree 官方 URDF 转换。
- **关节顺序映射**：IsaacSim 和 MuJoCo 的关节枚举顺序可能不同，部署前需对齐。

### 其他

- 训练日志中 `terrain_generator seed is not set`：可复现性提醒，建议后续固定 terrain seed
- `Gym has been unmaintained`：兼容性提醒，当前不阻塞训练
- `evaluate.py:146` checkpoint 正则仅匹配 `model_数字.pt`，不兼容 `model_last.pt` 等格式

---

## 训练计划（三任务并行，均独立）

### Teacher — 基线对照
```bash
PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/train.py \
  --task go2-train-teacher-rough \
  --num_envs 8192 --headless --device cuda:0 \
  --max_iterations 1500 \
  --logger wandb --group "Go2_Teacher"
```

### HIM — 基线对照
```bash
PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/train.py \
  --task go2-train-him-rough \
  --num_envs 8192 --headless --device cuda:0 \
  --max_iterations 1500 \
  --logger wandb --group "Go2_HIM"
```

### TAR RNN — 主实验
```bash
PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/train.py \
  --task go2-train-tar-rnn-rough \
  --num_envs 8192 --headless --device cuda:0 \
  --max_iterations 20000 \
  --logger wandb --group "Go2_TAR_RNN"
```

单卡按用时排序：先跑 Teacher、HIM（各 1500 iter），最后跑 TAR RNN（20000 iter）。

---

## 后续计划

1. 完成三个任务训练，记录 best/final checkpoint
2. 执行 Go2 的 ID/OOD 评估并补齐指标表
3. 编写 TAR RNN 专用 JIT 导出器
4. 准备 Go2 MuJoCo XML 并验证关节顺序映射
5. MuJoCo sim 验证后部署到 Go2 实机

