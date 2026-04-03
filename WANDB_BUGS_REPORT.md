# W&B 代码 Bug 报告

## 🔴 严重 Bug（会导致运行失败）

### Bug #1：在 WandbSummaryWriter 中访问不存在的变量

**位置**: `exts/tarloco/utils/logger.py` 第 350-351 行

**问题代码**:
```python
args_cli = getattr(locs["self"], "args_cli", None)
note = getattr(args_cli, "note", "") or locs["self"].cfg.get("note", "") or ""
```

**问题**:
1. `locs` 字典来自 `locals()` 调用
2. 在 `on_policy_runner.py` 第 202 行调用时：
   ```python
   self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, locs=locals())
   ```
   此时 `locs["self"]` 是 `OnPolicyRunner` 实例，不是 `LoggerWrapper`
3. `OnPolicyRunner` 实例可能没有 `args_cli` 属性
4. 如果 `args_cli` 不存在或为 None，下一行会因为对 None 调用 `.note` 而失败

**症状**:
- AttributeError: 'NoneType' object has no attribute 'note'
- 或 AttributeError: 'OnPolicyRunner' object has no attribute 'args_cli'

**修复**:
```python
# 改为
args_cli = getattr(locs["self"], "args_cli", None)
note = (getattr(args_cli, "note", "") if args_cli else "") or locs["self"].cfg.get("note", "") or ""
```

---

### Bug #2："args_cli" 变量在第 353 行被访问但在第 350 行可能为 None

**位置**: `exts/tarloco/utils/logger.py` 第 353-354 行

**问题代码**:
```python
group = getattr(args_cli, "group", None) or locs["self"].cfg.get("group", None)
job_type = getattr(args_cli, "job_type", None) or locs["self"].cfg.get("job_type", None)
```

**问题**:
- 同样的问题：`args_cli` 可能是 None
- 这些行假设了 `args_cli` 已经被正确地从某处获取

**修复**:
```python
group = (getattr(args_cli, "group", None) if args_cli else "") or locs["self"].cfg.get("group", None)
job_type = (getattr(args_cli, "job_type", None) if args_cli else "") or locs["self"].cfg.get("job_type", None)
```

---

### Bug #3：entity 为 None 时仍然尝试使用它

**位置**: `exts/tarloco/utils/logger.py` 第 360-361 行

**问题代码**:
```python
if wandb_continue_run and entity:
    continue_id = self.find_run_id_by_name(path=f"{entity}/{project}", run_name=wandb_continue_run)
```

**问题**:
- 如果 `entity` 是 None（环境变量 `WANDB_ENTITY` 未设置且配置中没有），代码会跳过
- 但稍后在第 365 行：`entity=entity,` 会传递 None 到 wandb.init()
- wandb.init() 可能因此出现问题或使用默认不正确的行为

**症状**:
- W&B 链接错误
- Run 无法正确上传到用户账户
- 数据保存到默认或公共项目

**修复**:
应该先检查用户已登录，然后获取默认 entity：
```python
# 在 wandb.init() 之前添加
if entity is None:
    # 获取已登录的用户
    try:
        api = wandb.Api()
        entity = api.default_entity
    except:
        print("[WARNING] Could not determine wandb entity. Set WANDB_ENTITY env var")
```

---

## 🟡 中等 Bug（可能导致不完整的日志）

### Bug #4：在 LoggerWrapper 中无法正确处理 `args_cli`

**位置**: `exts/tarloco/utils/logger.py` 第 80 行

**问题代码**:
```python
self.summary_writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, locs=locals())
```

**问题**:
- 传递 `locals()` 作为参数不是最佳实践
- 在不同的调用上下文中，`locs["self"]` 会指向不同的对象
- 代码难以维护和调试

**推荐修复**:
```python
# 直接传递所需的参数，而不是 locals()
self.summary_writer = WandbSummaryWriter(
    log_dir=self.log_dir,
    flush_secs=10,
    cfg=self.cfg,  # 或者具体的配置字典
    args_cli=self.args_cli
)

# 然后修改 WandbSummaryWriter.__init__ 签名：
def __init__(self, log_dir: str, flush_secs: int, cfg: dict, args_cli=None):
    # ... 使用 cfg 和 args_cli 直接
```

---

### Bug #5：video_path 可能包含相对路径，导致上传失败

**位置**: `exts/tarloco/utils/logger.py` 第 275 行

**问题代码**:
```python
for idx, video_path in enumerate(matching_videos):
    print(f"[INFO]: Matching video to upload to wandb: {video_path}")
    wandb.log({
        f"Robot {self.robot_idx}": wandb.Video(
            video_path,  # ← 可能是相对路径）
            caption=os.path.basename(video_path),
            format="mp4",
        )
    })
```

**问题**:
- 当 `path is None` 时（第 269 行），代码使用 `video_folder_path` 中的文件名
- 但这些文件名是相对路径，可能在 wandb 上传时找不到文件

**症状**:
- 视频上传到 W&B 失败
- W&B 显示"File not found"错误

**修复**:
```python
# 在第 275 行之前添加绝对路径转换
if not os.path.isabs(video_path):
    if path is None:
        video_path = os.path.join(video_folder_path, video_path)
```

---

### Bug #6：config 字典可能包含不可序列化的对象

**位置**: `exts/tarloco/utils/logger.py` 第 404 行

**问题代码**:
```python
def store_config(self, **configs):
    for key, config in configs.items():
        wandb.config.update({key: config})
```

**问题**:
- config 中可能包含不可 JSON 序列化的对象（如 tensor、函数等）
- wandb 会无法序列化这些对象

**症状**:
- "Object of type X is not JSON serializable"
- wandb.config.update() 失败或只记录了部分配置

**修复**:
```python
def store_config(self, **configs):
    for key, config in configs.items():
        if isinstance(config, dict):
            # 递归转换字典
            sanitized_config = self._make_json_serializable(config)
            wandb.config.update({key: sanitized_config})
        else:
            wandb.config.update({key: str(config)})

def _make_json_serializable(self, obj):
    """将对象转换为可 JSON 序列化的格式"""
    if isinstance(obj, dict):
        return {k: self._make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [self._make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.numel() < 100 else f"tensor_shape_{obj.shape}"
    else:
        try:
            # 尝试进行 JSON 编码测试
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
```

---

## 🟢 轻微 Bug（不影响功能，但可以改进）

### Bug #7：torch.cat() 持续增加内存

**位置**: `exts/tarloco/utils/logger.py` 第 148-153 行等

**问题代码**:
```python
self.signal_logs["contact_forces_z"] = torch.cat(
    (
        self.signal_logs["contact_forces_z"],
        signals["contact_forces_z"].unsqueeze(0),
    ),
    dim=0,
)
```

**问题**:
- 评估期间持续 cat 张量会导致内存不断增长
- 对于长时间的评估，可能导致 OOM

**优化**:
使用列表而不是持续 cat：
```python
# __init__ 中改为
self.signal_logs = {
    "contact_forces_z": [],
    "base_vel_x_y_yaw": [],
    # ...
}

# log_info 中改为
self.signal_logs["contact_forces_z"].append(signals["contact_forces_z"].cpu())

# upload_logs 中改为
contact_forces_tensor = torch.stack(self.signal_logs["contact_forces_z"]).to(device)
```

---

### Bug #8：find_run_id_by_name 可能很慢

**位置**: `exts/tarloco/utils/logger.py` 第 525-531 行

**问题代码**:
```python
@staticmethod
def find_run_id_by_name(path, run_name):
    api = wandb.Api()
    runs = api.runs(path)  # ← 获取所有 runs，可能很慢
    for run in runs:
        if run.name == run_name:
            return run.id
    return None
```

**问题**:
- 如果项目有许多 runs，这会非常慢
- API 调用可能超时

**优化**:
```python
@staticmethod
def find_run_id_by_name(path, run_name):
    api = wandb.Api()
    # 使用过滤器获取更少的 runs
    runs = api.runs(path, filters={"display_name": run_name})
    for run in runs:
        if run.name == run_name:
            return run.id
    return None
```

---

## 📋 修复优先级

| Bug # | 优先级 | 影响 | 修复时间 |
|-------|--------|------|---------|
| #1 | 🔴 严重 | 训练时崩溃 | 10 分钟 |
| #2 | 🔴 严重 | 训练时崩溃 | 10 分钟 |
| #3 | 🔴 严重 | 日志无法同步 | 15 分钟 |
| #4 | 🟡 中等 | 代码质量差 | 20 分钟 |
| #5 | 🟡 中等 | 视频不上传 | 15 分钟 |
| #6 | 🟡 中等 | 配置丢失 | 20 分钟 |
| #7 | 🟢 轻微 | 内存泄漏 | 15 分钟 |
| #8 | 🟢 轻微 | 性能差 | 10 分钟 |

---

## 🔧 立即需要做什么

### 如果你现在就要训练（临时解决方案）

```bash
# 1. 设置 WANDB_ENTITY 环境变量
export WANDB_ENTITY=your-username

# 2. 运行训练时添加显式的 logger
python standalone/tarloco/train.py \
  --task go1-train-teacher-rough \
  --max_iterations 20000 \
  --headless \
  --logger wandb \
  --group 'TEACHER' \
  --note 'Test run' \
  --device cuda:0 \
  --seed 0

# 如果仍然失败，临时使用 tensorboard
python standalone/tarloco/train.py \
  --task go1-train-teacher-rough \
  --max_iterations 20000 \
  --headless \
  --device cuda:0 \
  --seed 0
```

### 长期解决方案

需要修复上述代码中的 bug。我可以提供完整的修复代码。

---

## 结论

最关键的三个 bug（#1、#2、#3）会阻止 W&B 正常工作。这些都来自于：
1. 使用 `locals()` 传递参数（不推荐）
2. 没有正确的错误处理和 None 检查
3. 混淆了在不同调用上下文中 `self` 的含义

**建议**：我可以现在就为你修复这些 bug。需要吗？
