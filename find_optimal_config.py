#!/usr/bin/env python3
"""
自动找到 RTX 4090 + 90GB RAM 的最优训练配置
"""

import torch
import psutil
import subprocess
import json
from pathlib import Path

def get_gpu_memory():
    """获取GPU显存信息（GB）"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return total, allocated, reserved
    return 0, 0, 0

def get_cpu_memory():
    """获取CPU内存信息（GB）"""
    mem = psutil.virtual_memory()
    total = mem.total / 1024**3
    available = mem.available / 1024**3
    used = mem.used / 1024**3
    return total, available, used

def estimate_batch_size_limit():
    """估算最大batch_size"""
    print("=" * 80)
    print("GPU 显存分析")
    print("=" * 80)

    total_mem, _, _ = get_gpu_memory()
    print(f"总显存: {total_mem:.2f} GB")

    # 估算模型和优化器占用
    # CLCRec 模型相对较小，大约 1-2GB
    model_memory = 2.0
    print(f"模型+优化器预估: ~{model_memory:.1f} GB")

    # 可用于batch的显存
    available_for_batch = total_mem - model_memory - 2.0  # 保留2GB buffer
    print(f"可用于batch: ~{available_for_batch:.1f} GB")

    # 每个样本的显存估算
    # user_tensor: (batch_size * 129,) * 4 bytes
    # item_tensor: (batch_size * 129,) * 4 bytes
    # embeddings: batch_size * 129 * 64 * 4 bytes
    # 混合精度会减半
    bytes_per_sample = (129 * 2 * 4 + 129 * 64 * 2) / 1024**3  # GB

    max_batch = int(available_for_batch / bytes_per_sample)

    print(f"\n每个样本估算显存: {bytes_per_sample * 1024:.2f} MB")
    print(f"理论最大batch_size: ~{max_batch}")

    # 保守估计，取80%
    safe_max = int(max_batch * 0.8)
    print(f"安全最大batch_size: ~{safe_max}")

    # 建议的测试值
    test_values = []
    for size in [256, 512, 1024, 2048, 4096]:
        if size <= safe_max:
            test_values.append(size)

    print(f"\n建议测试的batch_size: {test_values}")
    print()

    return test_values

def estimate_num_workers_limit():
    """估算最大num_workers"""
    print("=" * 80)
    print("CPU 内存分析")
    print("=" * 80)

    total_mem, available_mem, used_mem = get_cpu_memory()
    print(f"总内存: {total_mem:.2f} GB")
    print(f"已使用: {used_mem:.2f} GB")
    print(f"可用: {available_mem:.2f} GB")

    # 获取CPU核心数
    cpu_count = psutil.cpu_count(logical=False)
    logical_count = psutil.cpu_count(logical=True)
    print(f"物理核心: {cpu_count}")
    print(f"逻辑核心: {logical_count}")

    # 估算每个worker的内存占用
    # 主要是数据加载，movielens数据集较小
    # 每个worker大约需要 0.5-1GB
    memory_per_worker = 1.0  # GB

    # 基于内存的最大workers
    max_workers_by_memory = int((available_mem - 10) / memory_per_worker)  # 保留10GB

    # 基于CPU的推荐workers
    # 一般建议：4-8个workers，不超过物理核心数
    max_workers_by_cpu = min(cpu_count, 16)

    max_workers = min(max_workers_by_memory, max_workers_by_cpu)

    print(f"\n每个worker估算内存: ~{memory_per_worker:.1f} GB")
    print(f"基于内存的最大workers: {max_workers_by_memory}")
    print(f"基于CPU的推荐workers: {max_workers_by_cpu}")
    print(f"推荐最大workers: {max_workers}")

    # 建议的测试值
    test_values = []
    for size in [4, 8, 12, 16]:
        if size <= max_workers:
            test_values.append(size)

    print(f"\n建议测试的num_workers: {test_values}")
    print()

    return test_values

def estimate_optimal_config():
    """综合估算最优配置"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "RTX 4090 最优配置分析" + " " * 21 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    batch_sizes = estimate_batch_size_limit()
    num_workers_list = estimate_num_workers_limit()

    print("=" * 80)
    print("推荐配置")
    print("=" * 80)

    # 推荐配置
    if batch_sizes:
        recommended_batch = batch_sizes[-1] if len(batch_sizes) > 0 else 512
        if recommended_batch > 1024:
            recommended_batch = 1024  # 太大的batch可能影响收敛
    else:
        recommended_batch = 512

    if num_workers_list:
        recommended_workers = num_workers_list[-1] if len(num_workers_list) > 0 else 8
    else:
        recommended_workers = 8

    print(f"\n🎯 推荐配置：")
    print(f"   --batch_size={recommended_batch}")
    print(f"   --num_workers={recommended_workers}")

    print(f"\n📊 完整命令：")
    print(f"""
python main.py \\
  --batch_size={recommended_batch} \\
  --num_workers={recommended_workers} \\
  --l_r=0.001 \\
  --reg_weight=0.1 \\
  --num_neg=128 \\
  --has_a=True \\
  --has_t=True \\
  --has_v=True \\
  --lr_lambda=0.5 \\
  --temp_value=2.0 \\
  --num_sample=0.5
""")

    print("\n" + "=" * 80)
    print("配置组合测试建议")
    print("=" * 80)

    print("\n从保守到激进的测试方案：")

    configs = [
        {"batch": 256, "workers": 4, "level": "保守（当前）"},
        {"batch": 512, "workers": 8, "level": "推荐"},
        {"batch": 1024, "workers": 8, "level": "激进"},
        {"batch": 1024, "workers": 12, "level": "极限"}
    ]

    for i, cfg in enumerate(configs, 1):
        if cfg["batch"] in batch_sizes and cfg["workers"] in num_workers_list:
            print(f"\n{i}. {cfg['level']}配置:")
            print(f"   python main.py --batch_size={cfg['batch']} --num_workers={cfg['workers']} ...")

    print("\n" + "=" * 80)
    print("性能预估")
    print("=" * 80)

    print(f"""
当前速度: ~3400 it/s (batch_size=256, num_workers=4)

预期速度：
- batch_size=512, num_workers=8:  ~6000-7000 it/s  (1.8-2.0x)
- batch_size=1024, num_workers=8: ~8000-10000 it/s (2.4-3.0x)
- batch_size=1024, num_workers=12: ~10000-12000 it/s (3.0-3.5x)
""")

    print("\n" + "=" * 80)
    print("注意事项")
    print("=" * 80)
    print("""
1. 从推荐配置开始测试，如果没有OOM，可以尝试更大的batch
2. batch_size过大可能影响模型收敛，注意观察loss
3. num_workers过多会增加CPU和内存压力，观察系统负载
4. 如果遇到OOM：
   - 降低batch_size
   - 减少num_neg (从128降到64)
   - 检查是否有其他进程占用显存
5. 监控GPU利用率应该在90-100%，如果低于80%，增加batch_size
""")

    return recommended_batch, recommended_workers

def create_quick_test_script(batch_size, num_workers):
    """创建快速测试脚本"""
    script_path = Path("quick_test.sh")

    content = f"""#!/bin/bash
# 快速测试脚本 - 只跑1个epoch来测试速度

echo "=========================================="
echo "快速性能测试"
echo "batch_size={batch_size}, num_workers={num_workers}"
echo "=========================================="

python main.py \\
  --batch_size={batch_size} \\
  --num_workers={num_workers} \\
  --num_epoch=1 \\
  --l_r=0.001 \\
  --reg_weight=0.1 \\
  --num_neg=128 \\
  --has_a=True \\
  --has_t=True \\
  --has_v=True \\
  --lr_lambda=0.5 \\
  --temp_value=2.0 \\
  --num_sample=0.5

echo ""
echo "测试完成！检查上面的 it/s 速度"
"""

    script_path.write_text(content)
    script_path.chmod(0o755)

    print(f"\n✓ 已创建快速测试脚本: {script_path}")
    print(f"  运行: ./quick_test.sh")

if __name__ == "__main__":
    try:
        batch_size, num_workers = estimate_optimal_config()
        create_quick_test_script(batch_size, num_workers)

        print("\n" + "=" * 80)
        print("下一步")
        print("=" * 80)
        print("""
1. 先等当前训练epoch结束（或按 Ctrl+C 停止）
2. 运行快速测试: ./quick_test.sh
3. 观察速度提升，如果满意，使用推荐配置进行完整训练
4. 如果想要更激进的配置，手动修改 quick_test.sh 中的参数
""")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
