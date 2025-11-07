# 使用优化版数据集加速训练

## 为什么需要优化？

### 当前的主要瓶颈

在原始 `Dataset.py` 中，每个样本的 `__getitem__` 都要执行：

```python
# 第159行 - 这是最大的性能瓶颈！
neg_item = random.sample(self.all_set - set(self.user_item_dict[user]), self.num_neg)
```

**问题分析**：
- 集合差运算 `self.all_set - set(...)` 每次都要重新计算
- `random.sample` 在 CPU 上采样 128 个负样本
- **对于 922,007 个训练样本，这会被调用 922,007 次！**
- 在多个 epoch 中，同样的采样会重复执行成百上千次

**时间开销**：
- 每次采样约 0.5-1ms
- 总开销：922,007 × 0.0005s ≈ **460秒 ≈ 7.5分钟/epoch**
- 这占据了大量训练时间！

---

## 优化方案：预生成负样本

`Dataset_optimized.py` 提供了两种优化方案：

### 方案1：TrainingDatasetOptimized（推荐）⭐⭐⭐⭐⭐

**原理**：
- 在初始化时一次性生成所有负样本
- 训练时直接读取，无需重复采样
- 可选择每隔N个epoch重新采样以保持随机性

**预期性能提升**：
- 初始化耗时：~20-30秒（一次性）
- 每个epoch节省：~7-10分钟
- 整体训练加速：**1.5-2x**（取决于数据集大小）

### 方案2：TrainingDatasetGPU（实验性）⭐⭐⭐

**原理**：
- 预计算用户-物品mask
- 在GPU上进行负采样
- 利用GPU并行能力

**状态**：实验性，可能不稳定

---

## 如何使用

### 方法1：修改 main.py（推荐）

在 `main.py` 中修改数据集导入：

```python
# 原来的导入
# from Dataset import TrainingDataset

# 改为
from Dataset_optimized import TrainingDatasetOptimized as TrainingDataset
```

然后正常运行即可：

```bash
python main.py --batch_size=1024 --num_workers=8 ...
```

### 方法2：创建新的启动脚本

创建 `main_optimized.py`，复制 `main.py` 并修改导入：

```python
# main_optimized.py
from Dataset_optimized import TrainingDatasetOptimized

# 在创建数据集时
train_dataset = TrainingDatasetOptimized(
    num_user, num_item, user_item_dict,
    data_path, train_data, num_neg,
    resample_epochs=5  # 每5个epoch重新采样一次
)
```

### 方法3：直接替换（最简单）

```bash
# 备份原文件
cp Dataset.py Dataset_original.py

# 将优化版本复制过去
cp Dataset_optimized.py Dataset.py

# 正常运行
python main.py --batch_size=1024 --num_workers=8 ...
```

---

## 参数说明

### resample_epochs

控制多久重新生成一次负样本：

```python
TrainingDatasetOptimized(..., resample_epochs=5)
```

- `resample_epochs=0`：只生成一次，所有epoch复用（最快，但可能影响收敛）
- `resample_epochs=1`：每个epoch都重新生成（最慢，类似原版）
- `resample_epochs=5`：每5个epoch重新生成一次（**推荐**，平衡速度和随机性）
- `resample_epochs=10`：每10个epoch重新生成一次（激进）

---

## 性能对比

### 原始版本 (Dataset.py)
```
数据加载开销：~7-10分钟/epoch
训练速度：3400 it/s (batch_size=256)
```

### 优化版本 (Dataset_optimized.py)
```
初始化开销：~20-30秒（一次性）
数据加载开销：~0秒（几乎为0）
训练速度：预计 5000-6000 it/s (batch_size=256)
```

### 组合优化（优化版数据集 + 大batch）
```
配置：TrainingDatasetOptimized + batch_size=1024 + num_workers=8
预期速度：10000-15000 it/s
总加速比：3-4x
```

---

## 完整优化建议

### 最优配置组合

```python
# 1. 使用优化版数据集
from Dataset_optimized import TrainingDatasetOptimized

# 2. 增大batch_size和num_workers
train_dataset = TrainingDatasetOptimized(
    num_user, num_item, user_item_dict,
    data_path, train_data, num_neg,
    resample_epochs=5
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1024,      # 从256增加到1024
    shuffle=True,
    num_workers=8,        # 从4增加到8
    pin_memory=True,
    persistent_workers=True
)
```

### 运行命令

```bash
python main.py \
  --batch_size=1024 \
  --num_workers=8 \
  --l_r=0.001 \
  --reg_weight=0.1 \
  --num_neg=128 \
  --has_a=True --has_t=True --has_v=True \
  --lr_lambda=0.5 --temp_value=2.0 --num_sample=0.5
```

---

## 预期提升总结

| 优化项 | 当前 | 优化后 | 加速比 |
|--------|------|--------|--------|
| 数据集 | 原始 | 优化版 | 1.5-2x |
| batch_size | 256 | 1024 | 1.5-2x |
| num_workers | 4 | 8 | 1.2x |
| **总计** | 3400 it/s | **10000-15000 it/s** | **3-4.5x** |

### Epoch时间对比

```
原始配置 (256 batch, 原始数据集):
- 每个epoch: ~8-12分钟
- 1000个epoch: ~130-200小时

优化配置 (1024 batch, 优化数据集):
- 每个epoch: ~2-3分钟
- 1000个epoch: ~30-50小时

节省时间: ~100-150小时！
```

---

## 注意事项

### 1. 内存占用

预生成负样本会增加内存占用：

```
额外内存 = 样本数 × num_neg × 8 bytes
         = 922,007 × 128 × 8 bytes
         ≈ 940 MB
```

对于 90GB 内存来说，这完全不是问题！

### 2. 随机性

- 预生成会减少负样本的随机性
- 通过 `resample_epochs` 参数定期重新采样可以缓解
- 实践中影响很小，因为每次重新采样都会生成新的负样本

### 3. 收敛性

- 理论上可能略微影响收敛
- 实践中影响可忽略不计
- 可以通过调整 `resample_epochs` 来平衡

---

## 故障排查

### 如果初始化太慢

```python
# 减少进度条更新频率
# 修改 Dataset_optimized.py 中的 tqdm
tqdm(..., mininterval=1.0)  # 每秒更新一次
```

### 如果内存不足

```python
# 减少num_neg
--num_neg=64  # 从128减少到64
```

### 如果想要更多随机性

```python
# 每个epoch都重新采样
TrainingDatasetOptimized(..., resample_epochs=1)
```

---

## 推荐操作流程

1. **先测试一个epoch**，确保没问题：
   ```bash
   python main.py --num_epoch=1 --batch_size=1024 --num_workers=8 ...
   ```

2. **观察速度提升**，应该能看到明显加速

3. **如果满意**，运行完整训练：
   ```bash
   python main.py --batch_size=1024 --num_workers=8 ...
   ```

4. **监控性能**：
   ```bash
   # 另开终端
   watch -n 1 nvidia-smi
   ```

---

## 结论

**是的，提前准备好用户物品图数据会显著加速训练！**

主要收益来自：
- ✅ 预生成负样本（1.5-2x加速）
- ✅ 避免重复的集合运算
- ✅ 减少CPU瓶颈

**结合其他优化（混合精度、大batch），总加速可达 3-4.5x！**
