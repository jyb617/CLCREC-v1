# CLCRec 性能优化指南 (RTX 4090)

## 已应用的优化

### 1. 混合精度训练 (AMP) ⚡
**位置**: `Train.py`
**影响**: 高 (预计加速 1.5-2x)

```python
from torch.cuda.amp import autocast, GradScaler

# 使用混合精度前向传播
with autocast():
    loss, model_loss, reg_loss = model.loss(...)

# 使用 GradScaler 进行反向传播
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**效果**:
- 充分利用 RTX 4090 的 Tensor Core
- 减少显存占用
- 加速训练过程

### 2. DataLoader 优化 📦
**位置**: `main.py`
**影响**: 中等 (预计加速 1.2-1.5x)

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,           # 加速 CPU-GPU 传输
    persistent_workers=True    # 保持 worker 不被销毁
)
```

**效果**:
- `pin_memory=True`: 使用页锁定内存，加速数据传输
- `persistent_workers=True`: 避免每个 epoch 重新创建 worker

### 3. 移除不必要的 retain_graph 🧹
**位置**: `Train.py`
**影响**: 低 (减少内存占用)

```python
# 之前: loss.backward(retain_graph=True)
# 现在: scaler.scale(loss).backward()
```

**效果**:
- 减少内存占用
- 略微提升速度

### 4. cuDNN Benchmark 优化 🎯
**位置**: `main.py`
**影响**: 低-中等

```python
torch.backends.cudnn.benchmark = True        # 自动寻找最优算法
torch.backends.cudnn.deterministic = False   # 提高性能
```

**效果**:
- cuDNN 自动选择最快的卷积算法
- 适合固定输入大小的场景

**注意**: 如果需要完全可重复的结果，设置 `deterministic=True`，但会牺牲一些性能。

---

## 预期性能提升

| 优化项 | 预计加速 | 优先级 |
|--------|----------|--------|
| 混合精度训练 | 1.5-2x | 高 |
| DataLoader 优化 | 1.2-1.5x | 高 |
| cuDNN benchmark | 1.1-1.2x | 中 |
| 移除 retain_graph | 1.05x | 低 |
| **总计** | **2-4x** | - |

**之前速度**: ~3000 it/s
**预期速度**: **6000-12000 it/s**

---

## 进一步优化建议

### 1. 增大批量大小 📊
**当前**: `batch_size=256`
**建议**: `batch_size=512` 或 `batch_size=1024`

RTX 4090 有 24GB 显存，可以支持更大的批量。

```bash
# 尝试更大的批量
python main.py --batch_size=512 ...
python main.py --batch_size=1024 ...
```

### 2. 优化负采样（较复杂）🔄
**当前问题**: CPU 上随机采样 128 次/样本，速度慢

**方案 A - 预计算负样本**:
```python
# 在训练前预先生成所有负样本
# 空间换时间
```

**方案 B - GPU 上采样**:
```python
# 在 GPU 上使用 torch.randperm 进行采样
# 需要重构 Dataset.__getitem__
```

**方案 C - 缓存采样**:
```python
# 每隔 N 个 epoch 重新采样一次
# 其他时候复用之前的负样本
```

### 3. 使用 torch.compile (PyTorch 2.0+) 🚀
如果你的 PyTorch >= 2.0:

```python
# 在 main.py 中
model = torch.compile(model, mode='max-autotune')
```

预计额外加速 1.2-1.5x。

### 4. 使用梯度累积（显存够用可忽略）
如果想要更大的有效批量但显存不够：

```python
accumulation_steps = 4  # 累积 4 个 batch

for i, (user, item) in enumerate(dataloader):
    with autocast():
        loss = model.loss(user, item) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## 性能监控

### 使用 nvidia-smi 监控 GPU 利用率
```bash
watch -n 1 nvidia-smi
```

**目标**: GPU 利用率应该在 **90-100%**

如果利用率低于 80%，说明存在瓶颈：
- **低于 80%**: 数据加载或 CPU 计算是瓶颈
- **90-100%**: GPU 性能得到充分利用 ✓

### 使用 nsys 进行详细分析
```bash
# 安装 Nsight Systems
nsys profile -o output python main.py --num_epoch=1

# 在 Nsight Systems GUI 中查看 output.qdrep
```

---

## 运行建议

### 基本运行（使用优化后的代码）
```bash
python main.py \
  --model_name='CLCRec' \
  --l_r=0.001 \
  --reg_weight=0.1 \
  --num_workers=4 \
  --num_neg=128 \
  --batch_size=256 \
  --has_a=True \
  --has_t=True \
  --has_v=True \
  --lr_lambda=0.5 \
  --temp_value=2.0 \
  --num_sample=0.5
```

### 推荐运行（增大批量）
```bash
python main.py \
  --model_name='CLCRec' \
  --l_r=0.001 \
  --reg_weight=0.1 \
  --num_workers=4 \
  --num_neg=128 \
  --batch_size=512 \         # 增大到 512
  --has_a=True \
  --has_t=True \
  --has_v=True \
  --lr_lambda=0.5 \
  --temp_value=2.0 \
  --num_sample=0.5
```

---

## 故障排查

### 1. 如果出现 OOM (Out of Memory)
```python
# 减小批量大小
--batch_size=256  # 或 128

# 或减少 num_neg
--num_neg=64  # 从 128 减少到 64
```

### 2. 如果速度没有提升
检查：
- [ ] PyTorch 版本 >= 1.6 (支持 AMP)
- [ ] CUDA 版本是否匹配
- [ ] GPU 驱动是否最新
- [ ] 是否有其他进程占用 GPU

```bash
# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 检查 GPU 占用
nvidia-smi
```

### 3. 如果需要完全可重复的结果
```python
# 在 main.py 中修改
torch.backends.cudnn.benchmark = False      # 关闭 benchmark
torch.backends.cudnn.deterministic = True   # 启用 deterministic
```

**注意**: 这会牺牲一些性能。

---

## 性能对比

### 优化前
- 速度: ~3000 it/s
- GPU 利用率: 50-70%
- 显存占用: ~8GB
- Epoch 时间: ~5-10 分钟

### 优化后（预期）
- 速度: 6000-12000 it/s
- GPU 利用率: 90-100%
- 显存占用: ~10-15GB (取决于批量)
- Epoch 时间: ~2-4 分钟

---

## 总结

通过以上优化，RTX 4090 的性能应该能得到充分发挥：

✅ **已完成**:
- 混合精度训练 (AMP)
- DataLoader 优化 (pin_memory, persistent_workers)
- 移除不必要的 retain_graph
- 启用 cuDNN benchmark

📝 **建议尝试**:
- 增大批量大小到 512 或 1024
- 优化负采样策略（较复杂）
- 使用 torch.compile (PyTorch 2.0+)

🎯 **预期结果**:
从 ~3000 it/s 提升到 **6000-12000 it/s** (2-4倍加速)
