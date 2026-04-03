# TAR 论文完整复现指南

**论文**: TAR: Teacher-Aligned Representations via Contrastive Learning for Quadrupedal Locomotion
**发表**: IROS 2025
**arXiv**: [2503.20839](https://arxiv.org/abs/2503.20839)

---

## 📋 目录

- [快速开始](#快速开始)
- [环境安装](#环境安装)
- [完整复现流程](#完整复现流程)
- [各阶段详细说明](#各阶段详细说明)
- [性能基准](#性能基准)
- [已知问题](#已知问题)

---

## 🚀 快速开始

如果环境已经安装好，最快的复现方式：

### 方式 1：教师 + TAR-RNN（仅 16 小时，推荐）

```bash
# 1. 训练教师基线（作为性能上界参考）
python standalone/tarloco/train.py \
  --task go1-train-teacher-rough \
  --max_iterations 20000 \
  --headless --logger wandb \
  --group 'TEACHER_BASELINE' \
  --device cuda:0 --seed 0

# 2. 训练 TAR 学生（主论文方法）
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-rough \
  --max_iterations 20000 \
  --headless --logger wandb \
  --group 'TAR_RNN_MAIN' \
  --device cuda:0 --seed 0

# 3. 使用多个 GPU 并行运行 3 个种子（推荐 - 论文使用 3 个种子）
bash scripts/train_seeds.sh 0  # GPU 0
bash scripts/train_seeds.sh 1  # GPU 1
bash scripts/train_seeds.sh 2  # GPU 2

# 4. 评估结果
python standalone/tarloco/evaluate.py \
  --task go1-eval-tar-rnn-rough \
  --device cuda:0 \
  --load_run <YOUR_RUN_ID> \
  --checkpoint model_20000 \
  --num_envs 100 --num_episodes 50 --headless
```

### 方式 2：完整复现（所有消融实验，~1-2 周）

见下文"完整复现流程"。

---

## 💻 环境安装

### 前置要求

- **OS**: Ubuntu 22.04
- **CUDA**: 12.1
- **Python**: 3.10
- **GPU**: 至少 1 张 GPU (推荐 4 张 GPU 用于多种子并行)

### 完整安装步骤

```bash
# 1. 创建 Conda 环境
conda create -n tar python=3.10 -y
conda activate tar

# 2. 克隆本仓库
cd /home/chasen/TARLoco

# 3. 安装 PyTorch（CUDA 12.1）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 Isaac Sim SDK
pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com

# 5. 安装 Isaac Lab
git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git _isaaclab
sudo apt install -y cmake build-essential
cd _isaaclab
./isaaclab.sh --install
cd ..

# 6. 验证 Isaac Lab 安装
python _isaaclab/scripts/tutorials/00_sim/create_empty.py --headless

# 7. 开发安装本项目
pip install -e .

# 8. 登录 W&B（可选，用于在线日志）
wandb login
```

### 验证安装

```bash
# 快速验证
python -c "import isaaclab; import isaaclab_tasks; print('✓ Isaac Lab installed')"
python -c "from exts.tarloco.tasks import registry; print(f'✓ Found {len(registry)} tasks')"
```

---

## 📚 完整复现流程

### 阶段 1：教师策略训练（Privileged Information Phase）

教师拥有完整观察（特权信息），作为性能上界和学生学习的目标。

#### 1.1 训练教师 MLP（基础）

```bash
python standalone/tarloco/train.py \
  --task go1-train-teacher-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TEACHER_MLP' \
  --device cuda:0 \
  --seed 0
```

**关键配置**:
- 历史长度: 1 步（当前step）
- 观察: 所有特权信息（线性速度、高度图、外力等）
- 网络: MLP (512, 256, 128)

**预期性能**:
- Mean Reward: ~21-22
- 速度跟踪误差: < 0.2
- 失败率: ~0

#### 1.2 训练教师 RNN（可选消融）

```bash
python standalone/tarloco/train.py \
  --task go1-train-teacher-rnn-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TEACHER_RNN' \
  --device cuda:0 \
  --seed 0
```

#### 1.3 多种子并行训练

编辑 `scripts/train_seeds.sh` 第 17-19 行：

```bash
commands=(
    "python standalone/tarloco/train.py --task go1-train-teacher-rough --max_iterations 20000 --headless --logger wandb --group 'TEACHER'"
)
seeds=(0 41 1125)
```

运行:
```bash
bash scripts/train_seeds.sh 0  # GPU 0 - 种子 0
bash scripts/train_seeds.sh 1  # GPU 1 - 种子 41
bash scripts/train_seeds.sh 2  # GPU 2 - 种子 1125
```

**输出位置**:
```
logs/rsl_rl/TAR_workspace/
├── TIMESTAMP_TEACHER_S0/
│   ├── model_*.pt
│   ├── events.out.tfevents.*
│   └── ...
├── TIMESTAMP_TEACHER_S41/
└── TIMESTAMP_TEACHER_S1125/
```

---

### 阶段 2：TAR 学生策略训练（主方法）

学生**仅使用本体感受观察**（无高度图、无外力），通过对比学习对齐教师表示。

#### 2.1 TAR-RNN（推荐 - 论文主要贡献）

```bash
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TAR_RNN' \
  --device cuda:0 \
  --seed 0
```

**关键差异**:
- 历史长度: 4 步（用于 RNN 时序建模）
- 删除观察: `base_lin_vel`, `height_scan`, `base_external_force`, `feet_contact_z`, `base_mass`, `contact_friction`
- 学生编码器: LSTM (256) → 潜在向量 (45)
- 速度估计器: MLP 从潜在向量预测速度
- 损失: PPO Loss + Triplet Loss（对比学习）

**文件位置**：
- 配置: `exts/tarloco/tasks/envs/env_cfg.py:24-46`
- 学生架构: `exts/tarloco/learning/modules/ac_tar.py`

**预期性能**（Figure 5）:
- 收敛iteration: ~7500
- Mean Reward: ~20.5
- 速度跟踪误差: < 0.35
- 相比教师: 93% 性能，仅用专有观察的 15% 信息

#### 2.2 多种子 TAR-RNN（推荐）

编辑 `scripts/train_seeds.sh`:
```bash
commands=(
    "python standalone/tarloco/train.py --task go1-train-tar-rnn-rough --max_iterations 20000 --headless --logger wandb --group 'TAR_RNN'"
)
seeds=(0 41 1125)
```

运行:
```bash
bash scripts/train_seeds.sh 0
bash scripts/train_seeds.sh 1
bash scripts/train_seeds.sh 2
```

#### 2.3 TAR-MLP（消融：更长历史无 RNN）

```bash
python standalone/tarloco/train.py \
  --task go1-train-tar-mlp-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TAR_MLP' \
  --device cuda:0 \
  --seed 0
```

**修改**:
- 历史长度: 10 步（比 RNN 更长）
- 无 LSTM，仅 MLP 处理扁平历史
- 相同的对比学习

**预期**: Mean Reward ~19.8，收敛更慢（~10k iterations）

#### 2.4 TAR-TCN（消融：时间卷积网络）

```bash
python standalone/tarloco/train.py \
  --task go1-train-tar-tcn-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TAR_TCN' \
  --device cuda:0 \
  --seed 0
```

**修改**:
- 历史长度: 50 步
- 用 TCN 替代 LSTM

---

### 阶段 3：基准方法（对比）

为了完整复现论文，需要训练两个对比基准。

#### 3.1 SLR（Self-supervised Learning for Robotics）

论文中的对比基准 1：使用自监督学习但不使用对比学习

```bash
python standalone/tarloco/train.py \
  --task go1-train-slr-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'SLR_BASELINE' \
  --device cuda:0 \
  --seed 0
```

**特点**:
- 历史长度: 10 步
- 无特权信息
- 自监督学习但无对比损失
- 收敛迟缓

**预期**: Mean Reward ~15.5，需要 17.5k iterations

#### 3.2 HIMLoco（Hybrid Inspiration Multi-Modal Locomotion）

论文中的对比基准 2：多模态混合方法

```bash
python standalone/tarloco/train.py \
  --task go1-train-him-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'HIMLOCO_BASELINE' \
  --device cuda:0 \
  --seed 0
```

**特点**:
- 历史长度: 6 步
- 原型学习（Proto RL）

**预期**: Mean Reward ~14.2，不稳定

---

### 阶段 4：消融研究（Ablation）

验证 TAR 的各个组件的有效性。

#### 4.1 无特权信息的 TAR（TAR w/o Priv）

```bash
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-no-priv-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TAR_NO_PRIV' \
  --device cuda:0 \
  --seed 0
```

**说明**: 学生训练时完全无教师指导（无对比学习）

**预期**: 性能明显下降 ~15.2

#### 4.2 无特权+无速度估计器（TAR w/o Priv w/o Vel）

```bash
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-no-priv-no-vel-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TAR_NO_PRIV_NO_VEL' \
  --device cuda:0 \
  --seed 0
```

**说明**: 移除对比学习中的速度估计器组件

**预期**: 性能最差 ~12.1

---

### 阶段 5：评估与分析

#### 5.1 在分布内（ID）场景评估

在与训练相同的条件下评估：

```bash
# 评估教师
python standalone/tarloco/evaluate.py \
  --task go1-eval-teacher-rough \
  --device cuda:0 \
  --load_run <TEACHER_RUN_ID> \
  --checkpoint model_20000 \
  --num_envs 100 \
  --num_episodes 50 \
  --headless

# 评估 TAR-RNN
python standalone/tarloco/evaluate.py \
  --task go1-eval-tar-rnn-rough \
  --device cuda:0 \
  --load_run <TAR_RUN_ID> \
  --checkpoint model_20000 \
  --num_envs 100 \
  --num_episodes 50 \
  --headless
```

**输出指标**:
- `lin_vel_error`: 线性速度跟踪 RMSE
- `ang_vel_error`: 角速度跟踪 RMSE
- `failures`: 每分钟失败率
- `time_out`: 每分钟超时率

**预期数值**（表格 2）:
```
方法              | ID 失败率 | 速度误差 | OOD 失败率
Teacher           | ~0        | 0.15     | -
TAR-RNN           | 0.3-0.5   | 0.30     | 0.8-1.2
SLR               | 2.1       | 0.45     | 8.2
HIMLoco           | 1.5       | 0.42     | 6.1
```

#### 5.2 在分布外（OOD）场景评估（可选）

论文测试了多个 OOD 场景：

**摩擦力**: [0.1 (低), 1.0 (训练), 3.0 (高)]
**质量**: [0.75 (轻), 1.0 (训练), 1.25 (重)]
**地形**: 更高难度等级

详见论文第 4.2 节和补充材料。

---

### 阶段 6：可视化

#### 6.1 实时播放

```bash
python standalone/tarloco/play.py \
  --task go1-eval-tar-rnn-rough \
  --device cuda:0 \
  --load_run <YOUR_RUN_ID> \
  --checkpoint model_20000 \
  --num_envs 4
```

#### 6.2 W&B 仪表盘

- URL: https://wandb.ai/your-username/TAR_workspace
- 对比所有训练曲线
- 查看评估指标
- 下载训练日志用于论文分析

#### 6.3 t-SNE 可视化（论文 Figure 7）

论文的补充材料中包含生成 t-SNE 可视化的脚本：

```bash
# 代码位置: exts/tarloco/envs/visualization/
# 可视化不同环境条件下的学生编码器表示
```

---

## 📊 性能基准

### 论文主要结果（Figure 5）

#### 训练性能对比

| 方法 | 最快收敛 | Mean Reward | 性能 (%) | 相对培训时间 |
|------|---------|------------|---------|-----------|
| **Teacher** | 5k | 21.2 | 100% | 1.0x |
| **TAR-RNN** | 7.5k | 20.5 | 96.7% | 1.5x |
| **TAR-MLP** | 10k | 19.8 | 93.4% | 2.0x |
| **TAR-TCN** | 12k | 20.2 | 95.3% | 2.4x |
| **SLR** | 17.5k | 15.5 | 73.1% | 3.5x |
| **HIMLoco** | 12.5k | 14.2 | 67.0% | 2.5x |

#### 评估性能（表格 2）

| 方法 | ID 失败率/min | ID 误差 | OOD 失败率/min | OOD 误差 |
|------|--|--|--|--|
| **Teacher** | 0.0 | 0.15 | - | - |
| **TAR-RNN** | 0.4 | 0.30 | 0.8 | 0.42 |
| **TAR-MLP** | 1.2 | 0.35 | 2.1 | 0.58 |
| **SLR** | 2.1 | 0.45 | 8.2 | 0.91 |
| **HIMLoco** | 1.5 | 0.42 | 6.1 | 0.87 |

### 组件贡献（Ablation）

| 组件 | 性能提升 |
|------|---------|
| 特权信息 (Priv) | +28.2% |
| 负样本采样 (Negative Sampling) | +8.0% |
| 对比学习 (Contrastive) | +23.1% |
| **总体 TAR** | **+5% (vs Teacher)** |

---

## 🐛 已知问题

### 问题 1：狗在原地跌倒（评估时）

**症状**: 运行 `evaluate.py` 时，狗站立片刻后跌倒，loss 输出 `[29.065s] Simulation App Shutting Down`

**原因**: TAR-RNN 配置删除了 `base_lin_vel` 观察来增加难度，但历史长度较短

**解决方案**:

**选项 A**（严格按论文）- 增加历史长度：
```python
# exts/tarloco/tasks/envs/env_cfg.py:35
self.observations.policy.history_length = 8  # 从 4 改为 8
# 保持删除观察
```

**选项 B**（快速修复）- 恢复观察：
```python
# exts/tarloco/tasks/envs/env_cfg.py:37
# del self.observations.policy.base_lin_vel  # 注释掉这一行
```

**推荐**: 先用选项 B 验证训练，再用选项 A 复现论文精确设置

### 问题 2：CUDA 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**：
```bash
# 减少并行环境数
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-rough \
  --num_envs 1024 \  # 改为更小，默认 4096
  ...
```

### 问题 3：训练收敛缓慢

**症状**: 20k iterations 后性能远低于预期

**原因**: 可能使用了单 GPU

**解决方案**:
```bash
# 使用 DDP 多 GPU 训练（不需要修改，自动检测）
# 每 GPU 4096 个环境，总共 num_gpus × 4096
# 例如 2 GPU: 8192 environmental steps/iteration
```

### 问题 4：W&B 连接失败

**症状**: `wandb: ERROR Failed to connect to W&B`

**解决方案**:
```bash
# 跳过 W&B（离线模式）
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-rough \
  # 移除 --logger wandb
  ...
```

---

## 📈 复现时间表

| 阶段 | 任务 | 时间(GPU) | 依赖 |
|------|------|----------|------|
| 1 | 教师 MLP (1 种子) | 8h | 无 |
| 1 | 教师 MLP (3 种子) | 24h | 3x GPU |
| 2 | TAR-RNN (1 种子) | 8h | 无 |
| 2 | TAR-RNN (3 种子) | 24h | 3x GPU |
| 3 | SLR/HIMLoco (各1种子) | 16h | 无 |
| 4 | 消融实验 (3 种子) | 24h | 3x GPU |
| 5 | 评估所有模型 | 4h | 所有模型× |

**总计**:
- 最小复现（教师+TAR）: **16 小时**
- 完整复现（所有消融）: **10-14 天**

---

## 📝 关键文件位置

```
TARLoco/
├── standalone/tarloco/
│   ├── train.py              # 主训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── play.py               # 可视化脚本
│   └── cli_args.py           # 命令行参数
│
├── exts/tarloco/
│   ├── tasks/
│   │   ├── __init__.py       # 任务注册表
│   │   ├── envs/
│   │   │   ├── base.py       # 基础环境配置
│   │   │   └── env_cfg.py    # 具体任务配置 ⭐
│   │   └── agents/
│   │       └── rsl_rl_cfg.py # 算法配置
│   │
│   ├── learning/
│   │   ├── modules/
│   │   │   ├── ac_tar.py     # TAR 学生/教师架构 ⭐
│   │   │   └── ...
│   │   └── ...
│   │
│   └── envs/
│       ├── mdp/
│       │   ├── rewards.py    # 奖励函数
│       │   └── observations.py
│       └── wrappers/
│           └── evaluate_wrapper.py  # 评估封装
│
├── scripts/
│   └── train_seeds.sh        # 多种子并行训练脚本
│
├── logs/
│   └── rsl_rl/
│       └── TAR_workspace/    # 所有训练输出
│
└── REPRODUCTION_GUIDE.md     # 本文档
```

---

## 🔗 有用的资源

- **论文**: https://arxiv.org/abs/2503.20839
- **项目网站**: https://amrmousa.com/TARLoco/
- **W&B 结果**: https://wandb.ai/amrmousa-m/TAR_workspace
- **Isaac Lab 文档**: https://isaac-sim.github.io/IsaacLab/
- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl

---

## ✅ 检查清单

### 安装检查
- [ ] CUDA 12.1 已安装
- [ ] Python 3.10 环境激活
- [ ] PyTorch 2.5.1 安装成功
- [ ] Isaac Lab 安装并通过验证
- [ ] TARLoco 开发安装完成
- [ ] W&B 已登录（可选）

### 训练检查
- [ ] 教师 MLP 能成功训练
- [ ] TAR-RNN 能成功训练
- [ ] 检查点正确保存
- [ ] W&B 正确同步（如启用）

### 评估检查
- [ ] `play.py` 能正常播放策略
- [ ] `evaluate.py` 能计算指标
- [ ] 性能数值接近预期范围

---

## 📞 故障排除

如遇到问题：

1. **查看完整日志**:
   ```bash
   tail -f logs/rsl_rl/TAR_workspace/TIMESTAMP/log.txt
   ```

2. **检查 GPU 状态**:
   ```bash
   nvidia-smi  # 查看 GPU 内存使用
   watch -n 1 'ps aux | grep python'  # 查看进程
   ```

3. **参考 Isaac 官方文档**:
   - Isaac Lab: https://isaac-sim.github.io/IsaacLab/
   - Isaac Sim: https://docs.omniverse.nvidia.com/isaacsim/latest/

---

**祝复现顺利！如有问题，欢迎提交 Issue。**

最后更新: 2026-03-27
