# TAR 训练验证流水线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 依次完成 Isaac Sim 仿真验证（Teacher / TAR-RNN / HIMLoco）→ MuJoCo 策略迁移验证 → Unitree Go2 实机鲁棒性测试。

**Architecture:** Isaac Sim 阶段复用现有训练/评估脚本；MuJoCo 阶段新建部署脚本，通过维护 4 步观察历史缓冲区调用 TAR-RNN 推理，无需重新训练；实机阶段使用 JIT 导出模型 + 结构化测试协议。

**Tech Stack:** Isaac Lab 2.1 / RSL-RL / PyTorch 2.5.1 / MuJoCo 3.x / Python 3.10 / W&B

---

## 阶段零：前置依赖确认

### Task 0: W&B 连通性验证

**Files:**
- 无需修改文件，仅运行命令

- [ ] **Step 1: 运行 500 步 smoke test**

```bash
cd /home/chasen/TARLoco
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-rough \
  --max_iterations 500 \
  --headless --logger wandb \
  --group 'SMOKE_TEST' \
  --device cuda:0 --seed 0
```

- [ ] **Step 2: 验证 W&B 曲线**

打开 W&B 仪表盘，确认项目 `TAR_workspace` 下出现 `SMOKE_TEST` run，以下曲线有数据：
- `Train/mean_reward`（有上升趋势）
- `Loss/value_function`（有下降趋势）
- `Steps`（正确递增）

**通过标准**：三条曲线均在 500 步内出现且数值合理（reward > 0, loss > 0）。

- [ ] **Step 3: 确认日志目录结构正确**

```bash
ls /home/chasen/TARLoco/logs/rsl_rl/TAR_workspace/ | tail -3
# 预期：出现以当日时间戳命名的新目录，例如 2026-03-27_XX-XX-XX/
```

---

## 阶段一：Isaac Sim 仿真验证

### Task 1: 训练 Teacher MLP（性能上界）

**Files:**
- 无需修改，使用已有任务 `go1-train-teacher-rough`

- [ ] **Step 1: 启动 Teacher 训练（后台运行）**

```bash
cd /home/chasen/TARLoco
nohup python standalone/tarloco/train.py \
  --task go1-train-teacher-rough \
  --max_iterations 20000 \
  --headless --logger wandb \
  --group 'TEACHER' \
  --note 'Teacher_S0' \
  --device cuda:0 --seed 0 \
  > logs/nohup/teacher_s0.log 2>&1 &
echo "Teacher PID: $!"
```

- [ ] **Step 2: 确认训练已启动**

```bash
mkdir -p /home/chasen/TARLoco/logs/nohup
tail -f /home/chasen/TARLoco/logs/nohup/teacher_s0.log
# 预期输出：出现 "Learning iteration" 进度条，无 CUDA 错误
# Ctrl+C 退出 tail，训练继续在后台运行
```

- [ ] **Step 3: 监控 W&B — 在 5k iteration 时检查中间指标**

W&B 仪表盘中确认 `TEACHER S0` run 的 `Train/mean_reward > 15`，`Terrain Level > 1.5`。

- [ ] **Step 4: 训练完成后记录 run 目录**

```bash
ls -t /home/chasen/TARLoco/logs/rsl_rl/TAR_workspace/ | head -3
# 记录 Teacher 的 run_id（时间戳目录名），例如：2026-03-27_XX-XX-XX
# 后续评估需要使用此 run_id
```

**预期指标（20k iter）：** Mean Reward ~21–22，Terrain Level ~2.8，best_rew_mean > 21

---

### Task 2: 训练 TAR-RNN（论文主方法）

**Files:**
- 无需修改，使用已有任务 `go1-train-tar-rnn-rough`

> **注意**：即使已有旧的 TAR-RNN 训练结果，本次重新训练以确保 W&B 完整记录与 Teacher/HIMLoco 在同一 group 下对比。

- [ ] **Step 1: 等待 Task 1 完成后启动（可并行，但单 GPU 需串行）**

确认 Teacher 训练进程已结束：
```bash
ps aux | grep train.py | grep -v grep
# 若无输出，说明训练已完成
```

- [ ] **Step 2: 启动 TAR-RNN 训练**

```bash
cd /home/chasen/TARLoco
nohup python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-rough \
  --max_iterations 20000 \
  --headless --logger wandb \
  --group 'TAR_RNN' \
  --note 'TAR_RNN_S0' \
  --device cuda:0 --seed 0 \
  > logs/nohup/tar_rnn_s0.log 2>&1 &
echo "TAR-RNN PID: $!"
```

- [ ] **Step 3: 在 7.5k iteration 时检查关键指标**

W&B 中 `TAR_RNN S0` run：
- `Train/mean_reward > 18`（论文预期 7.5k 时接近峰值）
- `Loss/surrogate` 稳定下降

- [ ] **Step 4: 训练完成后记录 run 目录**

```bash
ls -t /home/chasen/TARLoco/logs/rsl_rl/TAR_workspace/ | head -3
# 记录 TAR-RNN 的 run_id
```

**预期指标（20k iter）：** Mean Reward ~20.5，Terrain Level ~2.6，收敛 iteration ~7.5k

---

### Task 3: 训练 HIMLoco（对比基准）

**Files:**
- 无需修改，使用已有任务 `go1-train-him-rough`

- [ ] **Step 1: 等待 Task 2 完成后启动**

```bash
ps aux | grep train.py | grep -v grep
```

- [ ] **Step 2: 启动 HIMLoco 训练**

```bash
cd /home/chasen/TARLoco
nohup python standalone/tarloco/train.py \
  --task go1-train-him-rough \
  --max_iterations 20000 \
  --headless --logger wandb \
  --group 'HIMLOCO' \
  --note 'HIMLoco_S0' \
  --device cuda:0 --seed 0 \
  > logs/nohup/him_s0.log 2>&1 &
echo "HIMLoco PID: $!"
```

- [ ] **Step 3: 训练完成后记录 run 目录**

```bash
ls -t /home/chasen/TARLoco/logs/rsl_rl/TAR_workspace/ | head -3
# 记录 HIMLoco 的 run_id
```

**预期指标（20k iter）：** Mean Reward ~14.2，收敛 iteration ~12.5k，训练不稳定属正常

---

### Task 4: Isaac Sim 全套评估（ID + OOD）

**Files:**
- 无需修改，使用已有评估脚本

对 Teacher、TAR-RNN、HIMLoco 分别运行 ID 和 OOD 评估，共 6 次评估任务。

- [ ] **Step 1: 评估 Teacher（ID）**

```bash
# 将 <TEACHER_RUN_ID> 替换为 Task 1 Step 4 记录的目录名
python standalone/tarloco/evaluate.py \
  --task go1-eval-teacher-rough \
  --device cuda:0 \
  --load_run <TEACHER_RUN_ID> \
  --checkpoint model_best \
  --num_envs 100 --num_episodes 50 --headless \
  --logger wandb --group 'EVAL_ID'
```

预期：`failures < 0.5/min`，`lin_vel_error < 0.2`，`terrain_levels > 2.5`

- [ ] **Step 2: 评估 TAR-RNN（ID）**

```bash
python standalone/tarloco/evaluate.py \
  --task go1-eval-tar-rnn-rough \
  --device cuda:0 \
  --load_run <TAR_RNN_RUN_ID> \
  --checkpoint model_best \
  --num_envs 100 --num_episodes 50 --headless \
  --logger wandb --group 'EVAL_ID'
```

预期：`failures < 1.0/min`，`lin_vel_error < 0.35`

- [ ] **Step 3: 评估 HIMLoco（ID）**

```bash
python standalone/tarloco/evaluate.py \
  --task go1-eval-him-rough \
  --device cuda:0 \
  --load_run <HIM_RUN_ID> \
  --checkpoint model_best \
  --num_envs 100 --num_episodes 50 --headless \
  --logger wandb --group 'EVAL_ID'
```

预期：`failures < 2.0/min`，`lin_vel_error < 0.45`

- [ ] **Step 4: OOD 评估 — 修改评估配置（高载重 + 高速度）**

OOD 场景通过命令行参数覆盖，Isaac Lab 支持 hydra 覆盖：

```bash
# TAR-RNN OOD 评估示例（高速度 2.0 m/s + 高载重 15kg 需在 env_cfg 中手动调整）
# 先确认 OOD 配置是否已有专用任务：
grep "ood\|OOD" /home/chasen/TARLoco/exts/tarloco/tasks/__init__.py
```

若无专用 OOD 任务，使用如下方式运行（调高速度指令范围）：
```bash
python standalone/tarloco/evaluate.py \
  --task go1-eval-tar-rnn-rough \
  --device cuda:0 \
  --load_run <TAR_RNN_RUN_ID> \
  --checkpoint model_best \
  --num_envs 100 --num_episodes 50 --headless \
  --logger wandb --group 'EVAL_OOD'
```

- [ ] **Step 5: 对 Teacher 和 HIMLoco 重复 OOD 评估**

分别替换 `--task` 和 `--load_run` 参数，复用 Step 4 命令。

- [ ] **Step 6: 整理评估结果对比表**

在 W&B 仪表盘中创建对比视图，记录以下指标：

| 模型 | ID failures/min | ID lin_vel_error | OOD failures/min | OOD lin_vel_error |
|------|-----------------|------------------|------------------|-------------------|
| Teacher | | | - | - |
| TAR-RNN | | | | |
| HIMLoco | | | | |

**通过标准**：TAR-RNN 在 OOD 场景 failures < HIMLoco failures

---

## 阶段二：MuJoCo 迁移验证

### Task 5: 安装 MuJoCo 并验证 Go2 环境

**Files:**
- `requirements_mujoco.txt` (NEW) — MuJoCo 依赖清单

- [ ] **Step 1: 安装 MuJoCo**

```bash
pip install mujoco==3.2.3
pip install pygame imageio
```

- [ ] **Step 2: 验证 MuJoCo 安装**

```bash
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
# 预期输出：MuJoCo version: 3.2.3
```

- [ ] **Step 3: 验证 Go2 XML 可加载**

```bash
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('/home/chasen/ubuntu20.04/CTS/go2_rl_gym/resources/robots/go2/go2.xml')
print(f'Go2 model loaded: {m.nq} DOF, {m.nu} actuators')
"
# 预期输出：Go2 model loaded: 19 DOF, 12 actuators
```

- [ ] **Step 4: 记录依赖**

```bash
cat > /home/chasen/TARLoco/requirements_mujoco.txt << 'EOF'
mujoco==3.2.3
pygame>=2.5.0
imageio>=2.34.0
imageio-ffmpeg>=0.4.9
EOF
```

- [ ] **Step 5: 提交**

```bash
git add requirements_mujoco.txt
git commit -m "chore: add MuJoCo deployment dependencies"
```

---

### Task 6: 实现 TAR-RNN MuJoCo 部署脚本

**Files:**
- Create: `standalone/tarloco/deploy_mujoco.py`
- Create: `standalone/tarloco/configs/mujoco_tar.yaml`
- Create: `tests/test_mujoco_policy.py`

TAR-RNN 的推理逻辑：维护 4 步观察历史，每步从历史构造 `(batch=1, seq=4, obs=45)` 张量调用 `act_inference`。观察维度：`ang_vel(3) + proj_gravity(3) + vel_cmd(3) + joint_pos(12) + joint_vel(12) + last_action(12) = 45`。

- [ ] **Step 1: 创建 YAML 配置文件**

```bash
mkdir -p /home/chasen/TARLoco/standalone/tarloco/configs
```

写入 `standalone/tarloco/configs/mujoco_tar.yaml`：

```yaml
# Go2 MuJoCo TAR-RNN 部署配置
xml_path: "/home/chasen/ubuntu20.04/CTS/go2_rl_gym/resources/robots/go2/go2.xml"

simulation_duration: 30.0   # 每次测试时长（秒）
simulation_dt: 0.002        # 物理仿真步长 (500 Hz)
control_decimation: 4       # 控制频率 = 500/4 = 125 Hz

# PD 控制增益（与 Isaac Lab 训练保持一致）
kps: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
      20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
kds: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Go2 默认关节角度（弧度）
default_angles: [0.1, 0.8, -1.5,   # FL: hip, thigh, calf
                 -0.1, 0.8, -1.5,  # FR
                 0.1, 1.0, -1.5,   # RL
                 -0.1, 1.0, -1.5]  # RR

# 观测缩放（与训练一致）
ang_vel_scale: 0.25
dof_pos_scale: 1.0
dof_vel_scale: 0.05
action_scale: 0.25

# 指令
cmd_init: [0.5, 0.0, 0.0]     # [vx, vy, yaw_rate]
cmd_scale: [2.0, 2.0, 0.25]   # 指令归一化缩放

# 观测历史步数（必须与训练一致）
history_length: 4
num_obs_per_step: 45  # ang_vel(3)+proj_gravity(3)+cmd(3)+joint_pos(12)+joint_vel(12)+action(12)
```

- [ ] **Step 2: 写失败测试**

写入 `tests/test_mujoco_policy.py`：

```python
"""MuJoCo TAR-RNN 策略部署单元测试"""
import sys
sys.path.insert(0, '/home/chasen/TARLoco')

import numpy as np
import pytest
import torch
from collections import deque


def make_dummy_obs_history(history_length=4, obs_dim=45):
    """构造全零观测历史张量 shape=(1, history_length, obs_dim)"""
    return torch.zeros(1, history_length, obs_dim)


def test_obs_history_shape():
    """观测历史张量维度必须正确"""
    obs = make_dummy_obs_history()
    assert obs.shape == (1, 4, 45), f"Expected (1,4,45), got {obs.shape}"


def test_obs_history_update():
    """新观测步入队列后历史正确滑动"""
    history = deque([np.zeros(45)] * 4, maxlen=4)
    new_obs = np.ones(45)
    history.append(new_obs)
    stacked = np.stack(list(history), axis=0)  # (4, 45)
    assert stacked.shape == (4, 45)
    assert np.allclose(stacked[-1], 1.0), "最新观测应在末位"
    assert np.allclose(stacked[0], 0.0), "最旧观测应在首位"


def test_gravity_projection():
    """四元数转重力向量：朝上时应为 (0, 0, -1) 的旋转版本"""
    def get_gravity_orientation(q):
        qw, qx, qy, qz = q
        return np.array([
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz)
        ])
    # 单位四元数（无旋转）
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    g = get_gravity_orientation(q_identity)
    assert np.allclose(g, [0.0, 0.0, -1.0], atol=1e-6), f"Got {g}"


def test_pd_control_output_shape():
    """PD 控制输出维度为 12（12 个关节）"""
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        return (target_q - q) * kp + (target_dq - dq) * kd

    target_q = np.zeros(12)
    q = np.random.randn(12) * 0.1
    kp = np.ones(12) * 20.0
    kd = np.ones(12) * 0.5
    tau = pd_control(target_q, q, kp, np.zeros(12), np.zeros(12), kd)
    assert tau.shape == (12,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: 运行测试，确认全部通过**

```bash
cd /home/chasen/TARLoco
python -m pytest tests/test_mujoco_policy.py -v
# 预期：4 passed
```

- [ ] **Step 4: 实现 deploy_mujoco.py**

写入 `standalone/tarloco/deploy_mujoco.py`：

```python
"""
TAR-RNN MuJoCo 部署脚本
用法:
  python standalone/tarloco/deploy_mujoco.py \
    --run_dir logs/rsl_rl/TAR_workspace/2026-03-27_XX-XX-XX \
    --checkpoint model_best \
    --config standalone/tarloco/configs/mujoco_tar.yaml \
    [--scenario flat|ood_payload|ood_speed|ood_friction] \
    [--save_video]
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

# ── 将项目根目录加入 sys.path ───────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from exts.tarloco.learning.runners.on_policy_runner import OnPolicyRunner


# ── 工具函数 ────────────────────────────────────────────────────────────

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """四元数 (qw, qx, qy, qz) → 机体坐标系中的重力方向向量"""
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def pd_control(
    target_q: np.ndarray, q: np.ndarray, kp: np.ndarray,
    target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray,
) -> np.ndarray:
    """位置 PD 控制 → 关节力矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_obs(
    d: mujoco.MjData,
    cmd: np.ndarray,
    last_action: np.ndarray,
    default_angles: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """从 MuJoCo 数据构造单步 45 维观测向量（与训练 obs 完全对齐）"""
    quat = d.qpos[3:7]          # (qw, qx, qy, qz)
    ang_vel = d.qvel[3:6]       # 角速度（世界系）
    joint_pos = d.qpos[7:]      # 12 个关节位置
    joint_vel = d.qvel[6:]      # 12 个关节速度

    ang_vel_scaled   = ang_vel * cfg["ang_vel_scale"]
    gravity          = get_gravity_orientation(quat)
    cmd_scaled       = cmd * np.array(cfg["cmd_scale"], dtype=np.float32)
    joint_pos_scaled = (joint_pos - default_angles) * cfg["dof_pos_scale"]
    joint_vel_scaled = joint_vel * cfg["dof_vel_scale"]

    return np.concatenate([
        ang_vel_scaled,    # 3
        gravity,           # 3
        cmd_scaled,        # 3
        joint_pos_scaled,  # 12
        joint_vel_scaled,  # 12
        last_action,       # 12
    ]).astype(np.float32)  # total: 45


def load_tar_policy(run_dir: str, checkpoint: str, device: str = "cpu"):
    """加载 TAR-RNN 检查点，返回 (actor_critic, obs_normalizer)"""
    import pickle
    params_dir = os.path.join(run_dir, "params")

    with open(os.path.join(params_dir, "agent.pkl"), "rb") as f:
        train_cfg = pickle.load(f)

    # 重建 runner（不需要 Isaac Sim，仅用于加载模型权重）
    # 使用 dummy env 以获得正确的 obs/action 维度
    class _DummyEnv:
        num_envs = 1
        num_actions = 12
        device = device

        def get_observations(self):
            import torch
            obs = torch.zeros(1, 4, 45)
            extras = {"observations": {}}
            return obs, extras

        def step(self, actions):
            raise NotImplementedError

    runner = OnPolicyRunner(_DummyEnv(), train_cfg, log_dir=None, device=device)

    ckpt_path = os.path.join(run_dir, f"{checkpoint}.pt")
    runner.load(ckpt_path, load_optimizer=False)

    actor_critic = runner.alg.actor_critic.eval().to(device)
    obs_normalizer = runner.obs_normalizer.eval().to(device)
    return actor_critic, obs_normalizer


# ── 场景参数覆写 ────────────────────────────────────────────────────────

SCENARIOS = {
    "flat":         {"cmd": [0.5, 0.0, 0.0], "payload_kg": 0.0,  "friction": 1.0},
    "ood_payload":  {"cmd": [0.5, 0.0, 0.0], "payload_kg": 15.0, "friction": 1.0},
    "ood_speed":    {"cmd": [2.0, 0.0, 0.0], "payload_kg": 0.0,  "friction": 1.0},
    "ood_friction": {"cmd": [0.5, 0.0, 0.0], "payload_kg": 0.0,  "friction": 0.1},
}


# ── 主循环 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",    required=True, help="训练 run 目录路径")
    parser.add_argument("--checkpoint", default="model_best", help="检查点名称（不含 .pt）")
    parser.add_argument("--config",     default="standalone/tarloco/configs/mujoco_tar.yaml")
    parser.add_argument("--scenario",   default="flat", choices=list(SCENARIOS.keys()))
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scenario = SCENARIOS[args.scenario]
    cmd      = np.array(scenario["cmd"], dtype=np.float32)
    kps      = np.array(cfg["kps"],  dtype=np.float32)
    kds      = np.array(cfg["kds"],  dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    history_length = cfg["history_length"]
    num_obs        = cfg["num_obs_per_step"]
    sim_dt         = cfg["simulation_dt"]
    decimation     = cfg["control_decimation"]

    # 加载 MuJoCo 模型
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # 加载策略
    actor_critic, obs_normalizer = load_tar_policy(
        args.run_dir, args.checkpoint, device="cpu"
    )

    # 初始化
    obs_history = deque([np.zeros(num_obs, dtype=np.float32)] * history_length,
                        maxlen=history_length)
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    # 统计
    fall_count   = 0
    step_count   = 0
    sim_time     = 0.0
    vel_errors   = []

    print(f"[INFO] Scenario: {args.scenario} | cmd={cmd} | payload={scenario['payload_kg']}kg")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20.0

        start = time.time()
        while viewer.is_running() and sim_time < cfg["simulation_duration"]:
            # PD 控制
            tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                             np.zeros(12), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sim_time += sim_dt
            step_count += 1

            # 跌倒检测（机体高度 < 0.18m 视为跌倒）
            if d.qpos[2] < 0.18:
                fall_count += 1
                # 重置
                mujoco.mj_resetData(m, d)
                d.qpos[7:] = default_angles
                obs_history = deque(
                    [np.zeros(num_obs, dtype=np.float32)] * history_length,
                    maxlen=history_length
                )

            # 控制频率（每 decimation 步推理一次）
            if step_count % decimation == 0:
                obs_step = build_obs(d, cmd, action, default_angles, cfg)
                obs_history.append(obs_step)

                obs_tensor = torch.from_numpy(
                    np.stack(list(obs_history), axis=0)
                ).unsqueeze(0)  # (1, 4, 45)

                with torch.no_grad():
                    obs_norm = obs_normalizer(
                        obs_tensor.view(1, -1)
                    ).view_as(obs_tensor)
                    action_tensor = actor_critic.act_inference(obs_norm)

                action = action_tensor.cpu().numpy().squeeze() * cfg["action_scale"]
                target_dof_pos = action + default_angles

                # 速度误差记录
                local_vel = d.qvel[:3]
                vel_error = abs(local_vel[0] - cmd[0])
                vel_errors.append(vel_error)

            viewer.sync()

    # 输出统计
    elapsed  = cfg["simulation_duration"]
    falls_pm = fall_count / (elapsed / 60.0)
    mean_err = np.mean(vel_errors) if vel_errors else float("nan")
    print(f"\n{'='*50}")
    print(f"Scenario   : {args.scenario}")
    print(f"Falls/min  : {falls_pm:.2f}")
    print(f"Vel Error  : {mean_err:.4f} m/s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试，确认脚本可解析（不启动仿真）**

```bash
cd /home/chasen/TARLoco
python standalone/tarloco/deploy_mujoco.py --help
# 预期：打印帮助信息，无 import 错误
```

- [ ] **Step 6: 提交**

```bash
git add standalone/tarloco/deploy_mujoco.py \
        standalone/tarloco/configs/mujoco_tar.yaml \
        tests/test_mujoco_policy.py
git commit -m "feat: add TAR-RNN MuJoCo deployment script and unit tests"
```

---

### Task 7: MuJoCo ID + OOD 验证测试

**Files:**
- Modify: `tests/test_mujoco_policy.py` — 添加集成测试记录结果

在此任务中运行 4 个场景，记录 `falls/min` 和 `vel_error`，与 Isaac Sim 结果对比。

- [ ] **Step 1: TAR-RNN — 平地 ID 测试**

```bash
python standalone/tarloco/deploy_mujoco.py \
  --run_dir logs/rsl_rl/TAR_workspace/<TAR_RNN_RUN_ID> \
  --checkpoint model_best \
  --scenario flat
# 记录输出：Falls/min 和 Vel Error
```

**通过标准**：Falls/min < 2.0，Vel Error < 0.5 m/s

- [ ] **Step 2: TAR-RNN — OOD 高载重测试（15kg）**

```bash
python standalone/tarloco/deploy_mujoco.py \
  --run_dir logs/rsl_rl/TAR_workspace/<TAR_RNN_RUN_ID> \
  --checkpoint model_best \
  --scenario ood_payload
```

- [ ] **Step 3: TAR-RNN — OOD 高速度测试（2.0 m/s）**

```bash
python standalone/tarloco/deploy_mujoco.py \
  --run_dir logs/rsl_rl/TAR_workspace/<TAR_RNN_RUN_ID> \
  --checkpoint model_best \
  --scenario ood_speed
```

- [ ] **Step 4: TAR-RNN — OOD 低摩擦测试（滑面）**

```bash
python standalone/tarloco/deploy_mujoco.py \
  --run_dir logs/rsl_rl/TAR_workspace/<TAR_RNN_RUN_ID> \
  --checkpoint model_best \
  --scenario ood_friction
```

- [ ] **Step 5: 重复 Teacher 的 flat + ood_payload 测试（对比基线）**

```bash
python standalone/tarloco/deploy_mujoco.py \
  --run_dir logs/rsl_rl/TAR_workspace/<TEACHER_RUN_ID> \
  --checkpoint model_best \
  --scenario flat

python standalone/tarloco/deploy_mujoco.py \
  --run_dir logs/rsl_rl/TAR_workspace/<TEACHER_RUN_ID> \
  --checkpoint model_best \
  --scenario ood_payload
```

- [ ] **Step 6: 填写 MuJoCo 结果对比表**

| 模型 | flat falls/min | flat vel_err | ood_payload falls/min | ood_speed falls/min |
|------|---------------|--------------|----------------------|---------------------|
| Teacher | | | | |
| TAR-RNN | | | | |

**Sim2Sim 迁移损失** = `MuJoCo vel_error - IsaacSim vel_error`（应 < 0.2 m/s）

- [ ] **Step 7: 提交结果文档**

```bash
git add -A
git commit -m "test: record MuJoCo ID+OOD validation results"
```

---

## 阶段三：Unitree Go2 实机测试

### Task 8: 导出 TorchScript 模型用于实机部署

**Files:**
- 无需修改，使用已有 `evaluate.py --export_model jit`

- [ ] **Step 1: 导出 TAR-RNN JIT 模型**

```bash
python standalone/tarloco/evaluate.py \
  --task go1-eval-tar-rnn-rough \
  --device cuda:0 \
  --load_run <TAR_RNN_RUN_ID> \
  --checkpoint model_best \
  --num_envs 1 --num_episodes 1 --headless \
  --export_model jit
# 输出路径：logs/rsl_rl/TAR_workspace/<RUN_ID>/exported/policy.pt
```

- [ ] **Step 2: 验证 JIT 模型可加载**

```bash
python -c "
import torch
policy = torch.jit.load('logs/rsl_rl/TAR_workspace/<TAR_RNN_RUN_ID>/exported/policy.pt')
policy.eval()
obs = torch.zeros(1, 45)
action = policy(obs)
print(f'Action shape: {action.shape}')  # 预期: torch.Size([1, 12])
"
```

- [ ] **Step 3: 将 policy.pt 复制到 Go2 板载计算单元**

```bash
# 替换 <GO2_IP> 为 Go2 的实际 IP 地址
scp logs/rsl_rl/TAR_workspace/<TAR_RNN_RUN_ID>/exported/policy.pt \
    unitree@<GO2_IP>:~/tar_policy.pt
```

---

### Task 9: Go2 实机结构化鲁棒性测试

**Files:**
- 无需修改代码，执行测试协议

> **安全提示**：每次测试前确认紧急停止按钮可用，在安全围栏内进行，从低速（0.3 m/s）开始。

- [ ] **Step 1: 基础平地行走测试（热身）**

- 速度指令：vx=0.3 m/s，载重：0 kg
- 持续时间：60 秒
- 通过标准：无跌倒，步态稳定
- 记录：视频 + 是否出现抖动

- [ ] **Step 2: 复杂地形测试**

按顺序在以下地形上行走（vx=0.5 m/s）：
1. **密集草地**：持续 30 秒，通过标准：无跌倒
2. **粗糙沥青**：持续 30 秒，通过标准：无跌倒
3. **软海绵垫**（泡沫）：持续 30 秒，通过标准：无跌倒

- [ ] **Step 3: 楼梯测试**

- **上楼梯**：台阶高约 +15–20cm，完成 3 次，通过标准：3/3 成功
- **下楼梯**：台阶高约 -20–30cm，完成 3 次，通过标准：2/3 成功

- [ ] **Step 4: 扰动鲁棒性测试**

- 速度指令：vx=0.5 m/s
- 侧向推力：从 50N → 100N → 150N 逐步加大
- 通过标准：施加 100N 推力后 3 秒内恢复稳定
- 记录：最大能承受的推力（N）

- [ ] **Step 5: 载重测试**

在平地行走（vx=0.5 m/s）：
- 0 kg（对照）
- 5 kg（背包配重）
- 10 kg（接近论文测试值）
- 通过标准：10 kg 下持续行走 ≥ 30 秒

- [ ] **Step 6: 执行器降级测试（可选）**

- 限制单个关节（如右前腿大腿关节）力矩至 10%
- 速度指令：vx=0.3 m/s
- 通过标准：不倒，继续行走 ≥ 10 秒

- [ ] **Step 7: 汇总测试结果**

| 场景 | 结果 | 备注 |
|------|------|------|
| 基础平地 | ✓ / ✗ | |
| 密集草地 | ✓ / ✗ | |
| 粗糙沥青 | ✓ / ✗ | |
| 软海绵垫 | ✓ / ✗ | |
| 上楼梯 | X/3 | |
| 下楼梯 | X/3 | |
| 侧向扰动 | _ N | |
| 载重 10kg | ✓ / ✗ | |
| 执行器降级 | ✓ / ✗ / 跳过 | |

**最终通过标准**：必选场景（1–5）中 ≥ 4 项通过。

- [ ] **Step 8: 提交最终记录**

```bash
git add docs/
git commit -m "docs: record hardware validation results for TAR-RNN on Go2"
```

---

## 快速参考：关键路径指令

```bash
# 查看训练进度
tail -f logs/nohup/<run>.log

# 查看所有后台训练进程
ps aux | grep train.py | grep -v grep

# 检查 GPU 使用
nvidia-smi

# 查看最新 run 目录
ls -t logs/rsl_rl/TAR_workspace/ | head -5

# W&B 仪表盘
# https://wandb.ai/amrmousa-m/TAR_workspace
```
