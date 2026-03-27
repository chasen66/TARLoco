# TAR 训练验证计划设计文档

**日期**: 2026-03-27
**目标**: Isaac Sim 仿真验证 → MuJoCo 迁移验证 → Unitree Go2 实机测试

---

## 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | ASUS TUF Gaming RTX 5070 Ti OC 16GB（单卡） |
| CPU | Intel i5-13600KF |
| 主板 | ASUS TUF B760M |
| 机器人 | Unitree Go2 |

---

## 阶段一：Isaac Sim 仿真验证

### 前置：W&B 连通性验证

在正式训练前，运行 500 步 smoke test 确认 wandb 曲线正常推送。

```bash
python standalone/tarloco/train.py \
  --task go1-train-tar-rnn-rough \
  --max_iterations 500 \
  --headless --logger wandb \
  --group 'SMOKE_TEST' \
  --device cuda:0 --seed 0
```

**通过标准**：wandb 仪表盘能实时看到 `Train/mean_reward`、`Loss/*`、`Terrain Level` 曲线正常更新。

---

### 地形配置（所有模型共用）

训练使用 `ROUGH_TERRAINS_CFG`，包含 7 种地形，通过课程学习从简到难：

| 地形类型 | 比例 | 难度说明 |
|---------|------|---------|
| `pyramid_stairs` | 10% | 上楼梯，台阶高 5–18cm |
| `pyramid_stairs_inv` | 10% | 下楼梯，台阶高 5–18cm |
| `boxes` | 10% | 随机方块障碍，高 2.5–10cm |
| `random_rough` | 10% | 随机粗糙地面 |
| `random_tracks` | 50% | 铁轨地形（论文特有），轨道高 10–20cm |
| `hf_pyramid_slope` | 5% | 上坡，坡度 0–0.3 |
| `hf_pyramid_slope_inv` | 5% | 下坡，坡度 0–0.3 |

---

### 三模型串行训练计划（单 GPU，seed=0）

| 顺序 | 模型 | 任务名 | 预计时长 | 目的 |
|------|------|--------|---------|------|
| 1 | Teacher MLP | `go1-train-teacher-rough` | ~8h | 性能上界参考 |
| 2 | TAR-RNN | `go1-train-tar-rnn-rough` | ~8h | 论文主方法 |
| 3 | HIMLoco | `go1-train-him-rough` | ~8h | 对比基准 |

所有模型训练 20,000 iterations，wandb 实时监控。

**训练命令模板**：
```bash
python standalone/tarloco/train.py \
  --task <TASK_NAME> \
  --max_iterations 20000 \
  --headless --logger wandb \
  --group '<GROUP_NAME>' \
  --device cuda:0 --seed 0
```

**预期训练指标**：

| 模型 | 预期 Mean Reward | 收敛 Iteration |
|------|-----------------|---------------|
| Teacher | ~21–22 | ~5k |
| TAR-RNN | ~20.5 | ~7.5k |
| HIMLoco | ~14.2 | ~12.5k |

---

### 评估方案

训练完成后对三个模型做全套评估。

**ID 评估**（分布内）：
- 地形难度：level 1–5（训练范围内）
- 摩擦系数：[0.1, 1.0]
- 载重：[0, 7.5] kg
- 最大速度：1.0 m/s

**OOD 评估**（分布外）：
- 地形难度：level 6–9
- 载重：15 kg（超出训练范围）
- 最大速度：2.0 m/s

**评估命令模板**：
```bash
python standalone/tarloco/evaluate.py \
  --task <EVAL_TASK_NAME> \
  --device cuda:0 \
  --load_run <RUN_ID> \
  --checkpoint model_20000 \
  --num_envs 100 --num_episodes 50 --headless
```

**关键指标**：

| 指标 | 说明 |
|------|------|
| `lin_vel_error` | 线性速度跟踪 RMSE |
| `ang_vel_error` | 角速度跟踪 RMSE |
| `failures` | 跌倒次数/分钟 |
| `terrain_levels` | 平均地形难度等级 |

---

## 阶段二：MuJoCo 迁移验证

**目的**：加载 Isaac Sim 训练好的检查点，在 MuJoCo 的 Go2 物理引擎中测试 Sim2Sim 迁移能力，无需重新训练。

### 测试场景

**ID 场景**（与 Isaac Sim 评估对齐）：
- 平地 + 轻度坡面
- 摩擦系数在训练范围内
- 载重在训练范围内

**OOD 场景**：
- 高载重：15 kg
- 高速度：2.0 m/s
- 低摩擦（滑面）

### 对比指标

以 Isaac Sim 评估结果为基准，计算 MuJoCo 下的性能差值，量化 Sim2Sim 迁移损失：

```
迁移损失 = MuJoCo指标 - IsaacSim指标
```

---

## 阶段三：Unitree Go2 实机测试

对齐论文 Figure 1 的五类场景，进行结构化鲁棒性测试。

### 测试矩阵

| 场景 | 具体内容 | 通过标准 |
|------|---------|---------|
| 复杂地形行走 | 密集草地、粗糙沥青、软海绵垫 | 全程无跌倒 |
| 楼梯测试 | 上楼梯 +30cm，下楼梯 -60cm | 完成上下各 3 次 |
| 扰动鲁棒性 | 侧向推力，目标 ≥ 100N | 受力后 3s 内恢复稳定 |
| 载重测试 | 0 → 5 → 10 kg 逐步加载 | 10kg 下持续行走 ≥ 30s |
| 执行器降级（可选） | 单关节力矩限制至 10% | 能继续行走不倒 |

### 部署流程

1. 导出 ONNX / TorchScript 格式模型
2. 部署到 Go2 板载计算单元
3. 按测试矩阵逐项测试，录制视频

---

## 成功标准总结

| 阶段 | 核心成功标准 |
|------|------------|
| Isaac Sim | TAR-RNN 在 OOD 场景下 failures < HIMLoco；W&B 全程正常记录 |
| MuJoCo | TAR-RNN 在 MuJoCo 中速度误差 < 0.5，迁移损失可接受 |
| 实机 | 通过 4/5 场景测试（执行器降级为可选） |
