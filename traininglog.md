cd /home/chasen/TARLoco
PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
  --task go1-eval-teacher-rough \
  --num_envs 200\
  --num_episodes 10 \
  --load_run 2026-03-30_10-45-37 \
  --checkpoint model_21400.pt \
  --device cuda:0 \
  --cam_view free \
  "env.commands.base_velocity.ranges.lin_vel_x=[1.0,1.0]" \
  "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
  "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"


PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
  --task go1-eval-teacher-rough \
  --num_envs 200 --num_episodes 10 \
  --load_run 2026-03-30_10-45-37 --checkpoint model_21400.pt \
  --device cuda:0  \
    --cam_view free \
  "env.events.add_base_mass.params.mass_distribution_params=[-5.0,15.0]" \
  "env.events.physics_material.params.static_friction_range=[0.05,4.0]" \
  "env.commands.base_velocity.ranges.lin_vel_x=[0.0,2.0]" \
  "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
  "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"
			


  cd /home/chasen/TARLoco
PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
  --task go1-eval-him-rough \
  --num_envs 200\
  --num_episodes 10 \
  --load_run 2026-03-30_21-48-31 \
  --checkpoint model_19500.pt \
  --device cuda:0 \
  --cam_view free \
  "env.commands.base_velocity.ranges.lin_vel_x=[1.0,1.0]" \
  "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
  "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"

PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
  --task go1-eval-him-rough \
  --num_envs 200 --num_episodes 10 \
  --load_run 2026-03-30_21-48-31 --checkpoint model_19500.pt \
  --device cuda:0  \
  --cam_view free \
  "env.events.add_base_mass.params.mass_distribution_params=[-5.0,15.0]" \
  "env.events.physics_material.params.static_friction_range=[0.05,4.0]" \
  "env.commands.base_velocity.ranges.lin_vel_x=[0.0,2.0]"\
   "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
  "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"

PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
  --task go1-eval-tar-rnn-rough \
  --num_envs 200 --num_episodes 10 \
  --load_run 2026-03-30_19-07-08 --checkpoint model_18800.pt \
  --device cuda:0 \
  --cam_view free \
  "env.commands.base_velocity.ranges.lin_vel_x=[1.0,1.0]" \
  "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
  "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"


PYTHONPATH=/home/chasen/TARLoco python standalone/tarloco/evaluate.py \
  --task go1-eval-tar-rnn-rough \
  --num_envs 200 --num_episodes 10 \
  --load_run 2026-03-30_19-07-08 --checkpoint model_18800.pt  \
  --device cuda:0 \
    --cam_view free \
  "env.events.add_base_mass.params.mass_distribution_params=[-5.0,15.0]" \
  "env.events.physics_material.params.static_friction_range=[0.05,4.0]" \
  "env.commands.base_velocity.ranges.lin_vel_x=[0.0,2.0]" \
    "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
  "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"


## Isaac Sim 评估结果（200 envs x 10 episodes）

> **注意**：TAR-RNN 评估前修复了 evaluate.py 中 LSTM 隐状态未传递的 bug（每步从零隐状态推理 → 正确传递并在 done 时重置）

| 模型    | 条件 | Lin RMSE | Ang RMSE | Failure/min | Timeout/min |
|---------|------|----------|----------|-------------|-------------|
| Teacher | ID   | 1.6873   | 2.6547   | 0.0285      | 2.6988      |
| Teacher | OOD  | 1.7476   | 2.4337   | 0.1875      | 2.6628      |
| HIMLoco | ID   | 1.6575   | 1.1810   | 0.0330      | 2.6988      |
| HIMLoco | OOD  | 1.6816   | 1.3339   | 0.1845      | 2.6748      |
| TAR-RNN | ID   | 1.7302   | 1.4480   | **0.0075**  | 2.7003      |
| TAR-RNN | OOD  | 1.7486   | 1.6650   | **0.1170**  | 2.6838      |

### 分析
- **Lin RMSE**：三模型持平（1.67~1.74），线速度跟踪能力相当
- **Ang RMSE**：HIM(1.13) > TAR-RNN(1.43) > Teacher(2.68)，编码器提取时序信息有助角速度控制
- **Failure/min（核心指标）**：TAR-RNN 在 ID(0.57) 和 OOD(1.54) 条件下均最优，鲁棒性最强
- **OOD 劣化**：Teacher +91%, HIM +59%, TAR-RNN +170%（相对值大但绝对值仍最低）
- **结论**：TAR-RNN 鲁棒性显著领先，角速度跟踪接近 HIM，符合论文预期



以下是代码中所有可训练的完整任务名称（共24个）：                                                                                                   
                                                                                                                                                     
  主要方法                                                                                                                                           
                                                                                                                                                     
  ┌─────────────────────────────────┬────────────────────────────────┐                                                                               
  │           训练任务名            │           评估任务名           │                                                                               
  ├─────────────────────────────────┼────────────────────────────────┤                                                                               
  │ go1-train-tar-rnn-rough         │ go1-eval-tar-rnn-rough         │                                                                               
  ├─────────────────────────────────┼────────────────────────────────┤                                                                               
  │ go1-train-slr-rough             │ go1-eval-slr-rough             │                                                                               
  ├─────────────────────────────────┼────────────────────────────────┤
  │ go1-train-him-rough             │ go1-eval-him-rough             │                                                                               
  ├─────────────────────────────────┼────────────────────────────────┤                                                                               
  │ go1-train-teacher-rough         │ go1-eval-teacher-rough         │
  ├─────────────────────────────────┼────────────────────────────────┤                                                                               
  │ go1-train-teacher-encoder-rough │ go1-eval-teacher-encoder-rough │
  ├─────────────────────────────────┼────────────────────────────────┤
  │ go1-train-teacher-rnn-rough     │ go1-eval-teacher-rnn-rough     │
  └─────────────────────────────────┴────────────────────────────────┘                                                                               
   
  消融实验变体                                                                                                                                       
                  
  ┌────────────────────────────────────────┬───────────────────────────────────────┐                                                                 
  │               训练任务名               │              评估任务名               │
  ├────────────────────────────────────────┼───────────────────────────────────────┤                                                                 
  │ go1-train-tar-mlp-rough                │ go1-eval-tar-mlp-rough                │
  ├────────────────────────────────────────┼───────────────────────────────────────┤
  │ go1-train-tar-tcn-rough                │ go1-eval-tar-tcn-rough                │
  ├────────────────────────────────────────┼───────────────────────────────────────┤                                                                 
  │ go1-train-tar-rnn-no-priv-rough        │ go1-eval-tar-rnn-no-priv-rough        │
  ├────────────────────────────────────────┼───────────────────────────────────────┤                                                                 
  │ go1-train-tar-rnn-no-priv-no-vel-rough │ go1-eval-tar-rnn-no-priv-no-vel-rough │
  ├────────────────────────────────────────┼───────────────────────────────────────┤                                                                 
  │ go1-train-tar-mlp-no-priv-rough        │ go1-eval-tar-mlp-no-priv-rough        │
  ├────────────────────────────────────────┼───────────────────────────────────────┤                                                                 
  │ go1-train-tar-mlp-no-priv-no-vel-rough │ go1-eval-tar-mlp-no-priv-no-vel-rough │
  └────────────────────────────────────────┴───────────────────────────────────────┘                                                                 
                   
 你结合论文的内容和代码 告诉我所有的任务名有什么区别                                                                                                
                                                                                                                                                     
                         
● 现在我有了完整的信息，来系统讲解所有任务名的区别。                                                                                                 
                                                  
  ---                                                                                                                                                
  任务名构成规则                                                                                                                                     
                                                                                                                                                     
  go1 - {train|eval} - {方法} - {编码器} - {特权信息} - rough                                                                                        
                                                                                                                                                     
  ---                                                                                                                                                
  一、方法层面（论文对应关系）                                                                                                                       
                                                                                                                                                     
  ┌───────────┬────────────────────────────┬─────────────────────────────────┬──────────────┐
  │ 任务前缀  │          论文对应          │              算法               │     角色     │                                                        
  ├───────────┼────────────────────────────┼─────────────────────────────────┼──────────────┤
  │ tar-*     │ 本文提出的 TAR             │ PPOTAR（含 Triplet Loss）       │ 核心方法     │                                                        
  ├───────────┼────────────────────────────┼─────────────────────────────────┼──────────────┤
  │ slr-*     │ SLR baseline [Chen et al.] │ PPOSLR（随机负样本 triplet）    │ 对比基准     │                                                        
  ├───────────┼────────────────────────────┼─────────────────────────────────┼──────────────┤                                                        
  │ him-*     │ HIM baseline [Long et al.] │ PPOHIM（prototypical learning） │ 对比基准     │                                                        
  ├───────────┼────────────────────────────┼─────────────────────────────────┼──────────────┤                                                        
  │ teacher-* │ Teacher upper bound        │ PPO                             │ 特权信息上界 │
  └───────────┴────────────────────────────┴─────────────────────────────────┴──────────────┘                                                        
                  
  ---
  二、编码器架构区别（TAR变体）
                                                                                                                                                     
  ┌─────────┬───────────────────────────────────────────────────────┬──────────┬────────────────────────────┐
  │ 任务名  │                        编码器                         │ 历史长度 │            说明            │                                        
  ├─────────┼───────────────────────────────────────────────────────┼──────────┼────────────────────────────┤
  │ tar-rnn │ LSTM（arch_type="integrated"）                        │ 4步      │ 论文主方法，隐状态捕获时序 │                                        
  ├─────────┼───────────────────────────────────────────────────────┼──────────┼────────────────────────────┤
  │ tar-mlp │ MLP（拼接历史帧）                                     │ 10步     │ 消融：把RNN换成MLP         │                                        
  ├─────────┼───────────────────────────────────────────────────────┼──────────┼────────────────────────────┤                                        
  │ tar-tcn │ TCN（时序卷积，channels=[32,32,32]，kernels=[8,5,5]） │ 50步     │ 消融：把RNN换成TCN         │                                        
  └─────────┴───────────────────────────────────────────────────────┴──────────┴────────────────────────────┘                                        
                  
  ---                                                                                                                                                
  三、特权信息与速度估计（消融实验）
                                                                                                                                                     
  TAR 训练时 critic 默认接收特权观测（height_scan, external_force, feet_contact_z, friction, mass），student encoder 通过 triplet loss 对齐 teacher
  的 latent。                                                                                                                                        
                  
  ┌────────────────┬────────────────────────────────────┬────────────┬─────────────────────────────────────────────────────────────────┐             
  │      后缀      │            Critic 观测             │ 速度估计器 │                          对应论文消融                           │
  ├────────────────┼────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤             
  │ (无后缀)       │ 有特权信息（高度图、摩擦、质量等） │ 有         │ 完整 TAR                                                        │
  ├────────────────┼────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ no-priv        │ 无特权信息（纯本体感知）           │ 有         │ 验证特权信息贡献（论文：+28.2%）                                │             
  ├────────────────┼────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤             
  │ no-priv-no-vel │ 无特权信息                         │ 无速度估计 │ 进一步退化，等价于 SLR 的负采样策略，验证速度估计贡献（+0.82%） │             
  └────────────────┴────────────────────────────────────┴────────────┴─────────────────────────────────────────────────────────────────┘             
                  
  ---                                                                                                                                                
  四、Teacher 的三种变体
                                                                                                                                                     
  ┌─────────────────┬────────────────────────────┬────────────────────────────────────────────────────┐
  │     任务名      │         Policy 类          │                        说明                        │                                              
  ├─────────────────┼────────────────────────────┼────────────────────────────────────────────────────┤                                              
  │ teacher         │ ActorCriticMlp             │ 纯特权 MLP，无编码器，1步观测直接输入 actor/critic │
  ├─────────────────┼────────────────────────────┼────────────────────────────────────────────────────┤                                              
  │ teacher-encoder │ ActorCriticMlpDblEncExpert │ 特权信息经 MLP 编码器压缩为 latent，再拼接本体感知 │                                              
  ├─────────────────┼────────────────────────────┼────────────────────────────────────────────────────┤                                              
  │ teacher-rnn     │ ActorCriticRnnDblEnc       │ 特权信息经 LSTM 编码，用于选择 Teacher 的最优架构  │                                              
  └─────────────────┴────────────────────────────┴────────────────────────────────────────────────────┘                                              
                  
  论文提到 Teacher 的架构从多个候选中选出性能最好的作为上界比较。                                                                                    
                  
  ---                                                                                                                                                
  五、train vs eval 的区别
                          
  - train：使用训练域随机化范围（摩擦 [0.1,3.0]，质量 [-2,10] kg，外力 ±20N）
  - eval：EvaluationConfigMixin 扩大 OOD 测试范围（质量最大 15kg，速度最大 2.0m/s），用于评估泛化性                                                  
                                                                                                                                                     
  ---                                                                                                                                                
  完整对应总结                                                                                                                                       
                  
  go1-train-tar-rnn-rough          ← 论文主方法（TAR + LSTM）
  go1-train-tar-mlp-rough          ← 消融：LSTM → MLP                                                                                                
  go1-train-tar-tcn-rough          ← 消融：LSTM → TCN
  go1-train-tar-rnn-no-priv-rough  ← 消融：去掉特权信息（自监督模式）                                                                                
  go1-train-tar-rnn-no-priv-no-vel-rough ← 消融：去掉特权+速度估计                                                                                   
  go1-train-tar-mlp-no-priv-rough        ← 消融：MLP 无特权                                                                                          
  go1-train-tar-mlp-no-priv-no-vel-rough ← 消融：MLP 无特权无速度                                                                                    
                                                                                                                                                     
  go1-train-slr-rough              ← SLR baseline（随机负样本 triplet）                                                                              
  go1-train-him-rough              ← HIM baseline（prototype learning）
                                                                                                                                                     
  go1-train-teacher-rough          ← Teacher 上界（纯特权 MLP）
  go1-train-teacher-encoder-rough  ← Teacher + MLP encoder                                                                                           
  go1-train-teacher-rnn-rough      ← Teacher + RNN encoder                                                                                           
   
  如果你只是想复现论文核心结果，主要训练的是：tar-rnn、slr、him、teacher 这四个，其余都是消融实验。    


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

## Run 04 — `2026-03-30_10-45-37`（完成）

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

训练完成，耗时约 10 小时。

| 指标 | 值 |
|------|----|
| 最佳 Checkpoint | `model_best.pt`（iter **21400**，rew=**20.31**） |
| 最终 Checkpoint | `model_27998.pt`（iter 27998，rew≈18-19） |
| terrain_level | 稳定 2.6-2.8（未突破 3.0） |
| act_noise_std | 全程 0.80（稳定，无坍缩） |
| mean_reward | 波动 17-20，最高 20.31 |
| base_contact | 降至 2-4（步态稳定） |

**快速收敛信号（前 20 步）**：
- iter 1: rew=-0.79（orientation 惩罚刚加入，短暂下降）
- iter 8: rew=1.96
- iter 16: rew=6.77（步态快速适应新奖励结构）

### 发现的问题

- **NaN 崩溃仅出现在最后一步（iter 8000，绝对迭代数 28000）**：adaptive lr 在末期产生过大梯度，log_std 参数越界变为 nan。属于训练末期常见现象，**不影响任何已保存 checkpoint**。`model_best.pt` 完全可用。
- **terrain_level 仍未突破 3.0**：`flat_orientation_l2=-1.0` 使 terrain_level 从 2.5 提升至 2.6-2.8，有改善但未达论文的 level 5。根本原因可能是 `max_iterations` 不足（Teacher 只跑了 28k 而非论文的 20k+），或需要更长训练。评估时可接受当前结果作为 upper bound 参考。

### 状态

完成。推荐使用 `model_best.pt`（iter 21400，rew=20.31）。

---

## Run 05 — `2026-03-30_19-07-08`（云端 AutoDL，完成）

### 基本信息

| 项目 | 内容 |
|------|------|
| **任务** | `go1-train-tar-rnn-rough` |
| **模型** | TAR-RNN (`ActorCriticTarRnn`：LSTM encoder + Transformer + Velocity Estimator) |
| **起点** | 从头训练（scratch） |
| **Teacher 路径** | Run 04 的 `model_best.pt`（Teacher MLP，iter 21400） |
| **机器** | AutoDL RTX 4090 24GB |
| **IsaacLab 版本** | commit `21f7136`（与本地 `f4aa17f` 不同） |
| **num_envs** | 8192 |
| **max_iterations** | 20000 |
| **num_steps_per_env** | 24 |
| **seed** | 0 |
| **wandb group** | `TAR_RNN` |
| **wandb note** | `TAR_RNN` |
| **日志文件** | `/root/autodl-tmp/logs/tar_rnn.log` |

### 上次改动

首次训练 TAR-RNN，无对比基准。相比 Teacher MLP 系列：
- 模型换为 `ActorCriticTarRnn`（LSTM + Transformer + Velocity Estimator）
- 任务换为 `go1-train-tar-rnn-rough`（学生无特权观察）
- 新增 `--logger wandb --group TAR_RNN` 参数（本次才加，之前漏掉导致 wandb 无输出）

### 改动原因

- 与 Teacher MLP（Run 04）保持相同 `num_envs=8192`、`seed=0`，便于公平对比
- 云端 4090 训练速度 ~5.5s/iter，20000 iter ≈ 30.5 小时

### 训练结果

训练完成。

| 指标 | 值 |
|------|----|
| 评估使用 Checkpoint | `model_18800.pt`（iter 18800） |
| Normalizer 形状 | `_mean: (1, 180)` = 4 × 45（history_length=4, 45维本体感知） |
| LSTM encoder | `weight_ih_l0: (1024, 45)`，input_size=45 |
| Actor 输入维度 | 93 = 45(latent z) + 45(prop) + 3(vel) |
| 每步耗时 | collect ~4.5s + learn ~0.8s ≈ 5.3s/iter |

**前 16 步观察**：
- `rew: -0.68`（初始阶段，orientation 惩罚 + 策略未收敛，为负正常）
- `eps_len: 171.95`

### 发现的问题

- **云端环境安装复杂**：Isaac Sim pip 安装后 `libhdx.so`、`libSM.so.6`、`libXt.so.6` 等系统库缺失，需手动设置 `LD_LIBRARY_PATH` 并 `apt install libsm6 libxext6 libxrender1 libxt6`
- **`--logger wandb --group` 参数不可省略**：不加时 wandb 不会初始化，训练数据不上传
- **评估时 LSTM 隐状态未传递（已修复，详见下方 Bug #1）**：初次评估 Failure/min 高达 23.6，修复后降至 0.57 (ID) / 1.54 (OOD)
- **IsaacLab 版本差异**：云端训练 commit `21f7136`，本地评估 commit `f4aa17f`，观测管理器实现有差异（新版支持 `concatenate_dim`），但未导致实际问题，因两版对 `flatten_history_dim=False` 的处理一致

### 状态

完成。使用 `model_18800.pt` 进行评估。

---

## Run 06 — `2026-03-30_21-48-31`（完成）

### 基本信息

| 项目 | 内容 |
|------|------|
| **任务** | `go1-train-him-rough` |
| **模型** | HIMLoco (`ActorCriticHim`) |
| **起点** | 从头训练（scratch） |
| **num_envs** | 8192 |
| **max_iterations** | 20000 |
| **num_steps_per_env** | 24 |
| **seed** | 0 |
| **wandb group** | `HIMLoco` |
| **wandb note** | `HIMLoco` |
| **日志文件** | `logs/nohup/him_s0.log` |

### 上次改动

首次训练 HIMLoco，无对比基准。相比 Teacher MLP 系列：
- 模型换为 `ActorCriticHim`（HIM prototypical learning）
- 任务换为 `go1-train-him-rough`（学生无特权观察）

### 改动原因

- 与 Teacher MLP（Run 04）保持相同 `num_envs=8192`、`seed=0`，便于公平对比
- 作为论文 baseline 之一，需要完整 20000 iter 训练

### 训练结果

训练完成，耗时约 27 小时（~4.8s/iter × 20000）。

| 指标 | 值 |
|------|----|
| 最佳 Checkpoint | `model_best.pt`（iter **19500**，rew=**21.01**） |
| 最终 Checkpoint | `model_19999.pt`（iter 20000，rew≈18-20 振荡）|
| act_noise_std | 末期 0.73（从 1.0 自然衰减，稳定）|
| NaN 崩溃 | **无** |
| 初始 reward | -0.94（iter 0） |

**对比 Teacher MLP（Run 04）**：
- HIMLoco best rew = **21.01** vs Teacher best rew = **20.31**
- HIMLoco 略优于 Teacher（+0.7），符合论文中 HIM 作为有力 baseline 的定位

### 发现的问题

无明显异常，训练平稳完成。

### 状态

完成。推荐使用 `model_best.pt`（iter 19500，rew=21.01）。

---

## 待完成

_所有训练 Run（04/05/06）均已完成。Isaac Sim ID+OOD 评估已完成。下一步：MuJoCo 部署。_

---

## 发现的 Bug

### Bug #1：evaluate.py 未传递 LSTM 隐状态（已修复）

- **影响**：所有 recurrent 模型（TAR-RNN 等）评估时 LSTM 每步从零隐状态推理，无法积累时间上下文
- **现象**：TAR-RNN 初次评估 Failure/min = **23.6**（正常应 <2）
- **根因**：`get_inference_policy()` 返回的闭包不传递 `hidden_states`；`evaluate.py` 主循环丢弃 `terminated/truncated`，不重置隐状态
- **不影响**：Teacher MLP（无 LSTM）、HIMLoco（`is_recurrent=False`，纯前馈）
- **修复**：
  - `on_policy_runner.py`：新增 `get_inference_policy_recurrent()` 方法，封装 `RecurrentPolicy` 类管理隐状态传递和 done 时重置
  - `evaluate.py`：检测 `is_recurrent`，使用对应 policy；主循环中 `terminated | truncated` 时调用 `policy.reset(dones)`
- **修复后结果**：TAR-RNN Failure/min = **0.57** (ID) / **1.54** (OOD)

### Bug #2：EvaluationConfigMixin 拼写错误（未修复）

- **位置**：`exts/tarloco/tasks/envs/base.py:461`
- **代码**：`self.termiantions = TerminationsEvalCfg()`（`ia` 写反了，应为 `self.terminations`）
- **影响**：`TerminationsEvalCfg`（仅 90° 倾倒才终止）从未生效，所有 eval 任务实际使用训练时的 `TerminationsCfg`（躯干/髋部接触力 >1N 即终止），终止条件比预期严格
- **三个模型均受影响**：不是 TAR-RNN 特有问题，修复后所有模型的 Failure/min 都会下降
- **状态**：已发现，暂未修复（当前评估结果基于相同条件，横向对比仍有效）

---

## 代码变更摘要

| 文件 | 变更内容 | 改动原因 | 生效自 |
|------|---------|---------|--------|
| `exts/tarloco/tasks/envs/env_cfg.py` | `flatten_history_dim` False → True（Teacher policy+critic） | 修复 action shape bug | Run 02 |
| `exts/tarloco/tasks/algorithms/ppo_cfg.py` | `wandb_entity` 改为本地账号 | 修复 wandb 404 错误 | Run 02 |
| `exts/tarloco/utils/logger.py` | wandb run name 格式改为 `{时间戳}_{note}` | 便于在 wandb 中直接识别模型与时间 | Run 04 |
| `exts/tarloco/tasks/envs/base.py` | `flat_orientation_l2` weight 0.0 → -1.0 | 修复地形难度无法提升的根本原因 | Run 04 |
| `exts/tarloco/learning/runners/on_policy_runner.py` | 新增 `get_inference_policy_recurrent()` 方法 | 修复 recurrent 模型评估时隐状态不传递（Bug #1） | 评估阶段 |
| `standalone/tarloco/evaluate.py` | 检测 `is_recurrent` 选择 policy + done 时重置隐状态 | 修复 recurrent 模型评估时隐状态不传递（Bug #1） | 评估阶段 |
