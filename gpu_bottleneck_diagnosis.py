#!/usr/bin/env python3
"""
GPU利用率诊断工具
帮助找出为什么GPU使用率上不去
"""

import subprocess
import time
import sys

def check_gpu_utilization():
    """实时监控GPU利用率"""
    print("=" * 80)
    print("GPU利用率诊断")
    print("=" * 80)
    print("\n正在监控GPU状态（按Ctrl+C停止）...\n")

    try:
        while True:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                gpu_util = int(values[0])
                mem_util = int(values[1])
                mem_used = int(values[2])
                mem_total = int(values[3])
                temp = int(values[4])
                power = float(values[5])

                print(f"\r GPU: {gpu_util:3d}% | Mem: {mem_util:3d}% ({mem_used}/{mem_total}MB) | "
                      f"Temp: {temp}°C | Power: {power:.1f}W", end='', flush=True)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n监控停止")

def analyze_bottleneck():
    """分析GPU利用率低的可能原因"""
    print("\n" + "=" * 80)
    print("GPU利用率低的常见原因分析")
    print("=" * 80)

    print("""
当GPU利用率持续低于80%时，说明GPU在等待数据或计算负载不足。

常见原因及解决方案：

1. 数据加载瓶颈 ⭐⭐⭐⭐⭐ (最常见)
   症状：
   - GPU利用率波动，不稳定
   - CPU某些核心100%，但GPU只有50-70%

   原因：
   - DataLoader跟不上GPU速度
   - 每个batch的CPU负采样太慢

   解决：
   ✓ 增加num_workers (4 -> 8或12)
   ✓ 使用pin_memory=True (已应用)
   ✓ 使用优化版数据集（预生成负样本）← 最重要！

   命令：
   python main.py --num_workers=8 --batch_size=1024 ...

2. Batch Size太小 ⭐⭐⭐⭐
   症状：
   - GPU利用率稳定但很低（30-50%）
   - GPU显存占用很少（<8GB）

   原因：
   - batch_size=256对RTX 4090来说太小
   - GPU计算能力没有被充分利用

   解决：
   ✓ 增大batch_size到1024或2048

   命令：
   python main.py --batch_size=1024 ...

3. CPU计算瓶颈 ⭐⭐⭐⭐⭐
   症状：
   - 训练时CPU某些核心100%
   - 进度条更新很慢

   原因：
   - random.sample()在CPU上采样128个负样本
   - 集合运算 (all_set - set(...)) 很慢

   解决：
   ✓ 使用Dataset_optimized.py预生成负样本 ← 关键！

   修改main.py:
   from Dataset_optimized import TrainingDatasetOptimized as TrainingDataset

4. I/O瓶颈 ⭐⭐
   症状：
   - 硬盘读写灯常亮
   - iowait高

   解决：
   ✓ 数据已经加载到内存，通常不是问题
   ✓ 如果有问题，考虑使用SSD

5. 模型太小 ⭐
   症状：
   - GPU利用率稳定在某个低值
   - 增大batch也没用

   原因：
   - CLCRec模型本身较小

   解决：
   ✓ 这不是问题，只要训练能正常进行

6. 混合精度问题 ⭐
   症状：
   - 使用AMP后利用率反而降低

   解决：
   ✓ 确保PyTorch版本>=1.6
   ✓ 检查是否有类型不匹配错误

=============================================================================
诊断方法
=============================================================================

方法1：查看nvidia-smi
-----------------------
nvidia-smi dmon -s u -d 1

观察：
- GPU Utilization 应该在 90-100%
- 如果在 50-70%，说明有瓶颈

方法2：使用htop查看CPU
-----------------------
htop

观察：
- 如果某些核心100%，可能是数据加载瓶颈
- 增加num_workers

方法3：PyTorch Profiler
-----------------------
在Train.py中添加：
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # 训练代码
    ...

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

=============================================================================
当前配置的问题
=============================================================================

根据你的代码，主要问题是：

1. ❌ CPU负采样瓶颈 (最严重)
   位置: Dataset.py:159
   问题: random.sample() 每个样本调用一次
   影响: 占用大量CPU时间，GPU在等待

2. ❌ batch_size太小
   当前: 256
   建议: 1024或2048

3. ⚠️ num_workers可能不够
   当前: 4
   建议: 8或12

=============================================================================
解决方案优先级
=============================================================================

高优先级（立即修复）：
1. 使用优化版数据集 → 预计GPU利用率 +20-30%
2. 增大batch_size到1024 → 预计GPU利用率 +10-20%
3. 增加num_workers到8 → 预计GPU利用率 +5-10%

低优先级：
4. 优化模型计算（已经挺好了）

=============================================================================
快速测试方案
=============================================================================

测试1：只改num_workers
-----------------------
python main.py --num_workers=8 --batch_size=256 ...

预期：GPU利用率略微提升 (5-10%)

测试2：只改batch_size
-----------------------
python main.py --num_workers=4 --batch_size=1024 ...

预期：GPU利用率明显提升 (10-20%)

测试3：使用优化数据集
-----------------------
# 修改main.py第7行
from Dataset_optimized import TrainingDatasetOptimized as TrainingDataset

python main.py --num_workers=4 --batch_size=256 ...

预期：GPU利用率显著提升 (20-30%)

测试4：组合优化（推荐）
-----------------------
# 修改main.py使用优化数据集
from Dataset_optimized import TrainingDatasetOptimized as TrainingDataset

python main.py --num_workers=8 --batch_size=1024 ...

预期：GPU利用率 90-100% ✓

=============================================================================
""")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        check_gpu_utilization()
    else:
        analyze_bottleneck()
        print("\n要实时监控GPU利用率吗？运行：")
        print("  python gpu_bottleneck_diagnosis.py monitor")
        print("\n或直接运行：")
        print("  watch -n 1 nvidia-smi")
