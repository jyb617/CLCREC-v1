#!/usr/bin/env python3
"""
GPU Performance Check for CLCRec on RTX 4090
检查 GPU 性能并提供优化建议
"""

import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import sys

def check_gpu_info():
    """检查GPU基本信息"""
    print("=" * 80)
    print("GPU 信息检查")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！")
        return False

    print(f"✓ CUDA 可用")
    print(f"✓ GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    print(f"✓ PyTorch 版本: {torch.__version__}")
    print(f"✓ GPU 数量: {torch.cuda.device_count()}")

    # 检查显存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3

    print(f"✓ 总显存: {total_memory:.2f} GB")
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已保留: {reserved:.2f} GB")
    print()

    return True


def check_cudnn():
    """检查 cuDNN 配置"""
    print("=" * 80)
    print("cuDNN 配置检查")
    print("=" * 80)

    print(f"cuDNN 可用: {torch.backends.cudnn.enabled}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")

    if not torch.backends.cudnn.benchmark:
        print("\n⚠️  建议: 启用 torch.backends.cudnn.benchmark = True")
        print("   这会让 cuDNN 自动寻找最优的卷积算法")

    if torch.backends.cudnn.deterministic:
        print("\n⚠️  警告: deterministic=True 会降低性能")
        print("   如果不需要完全可重复的结果，建议关闭")
    print()


def benchmark_data_transfer():
    """测试 CPU-GPU 数据传输速度"""
    print("=" * 80)
    print("数据传输速度测试")
    print("=" * 80)

    # 模拟一个batch的数据
    batch_size = 256
    num_neg = 128
    data_cpu = torch.randint(0, 10000, (batch_size, num_neg + 1), dtype=torch.long)

    # 测试不使用 pin_memory
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        data_gpu = data_cpu.cuda()
        torch.cuda.synchronize()
    time_normal = time.time() - start

    # 测试使用 pin_memory
    data_cpu_pinned = data_cpu.pin_memory()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        data_gpu = data_cpu_pinned.cuda(non_blocking=True)
        torch.cuda.synchronize()
    time_pinned = time.time() - start

    print(f"不使用 pin_memory: {time_normal:.4f}s (100次传输)")
    print(f"使用 pin_memory: {time_pinned:.4f}s (100次传输)")
    print(f"加速比: {time_normal/time_pinned:.2f}x")

    if time_normal / time_pinned > 1.2:
        print("\n✓ 建议: 在 DataLoader 中启用 pin_memory=True")
    print()


def benchmark_sampling():
    """测试负采样速度"""
    print("=" * 80)
    print("负采样速度测试")
    print("=" * 80)

    import random
    num_item = 10000
    num_neg = 128
    user_items = set(range(100))
    all_items = set(range(num_item))

    # CPU 随机采样
    start = time.time()
    for _ in range(1000):
        neg_items = random.sample(all_items - user_items, num_neg)
    time_cpu = time.time() - start

    # GPU 随机采样
    all_items_tensor = torch.arange(num_item, device='cuda')
    start = time.time()
    torch.cuda.synchronize()
    for _ in range(1000):
        # 简化的GPU采样（实际实现会更复杂）
        perm = torch.randperm(num_item, device='cuda')[:num_neg]
        neg_items_gpu = all_items_tensor[perm]
        torch.cuda.synchronize()
    time_gpu = time.time() - start

    print(f"CPU 采样 (random.sample): {time_cpu:.4f}s (1000次)")
    print(f"GPU 采样 (torch.randperm): {time_gpu:.4f}s (1000次)")
    print(f"加速比: {time_cpu/time_gpu:.2f}x")

    print("\n✓ 建议: 将负采样移到 GPU 上或使用预采样策略")
    print()


def benchmark_mixed_precision():
    """测试混合精度训练加速"""
    print("=" * 80)
    print("混合精度训练测试")
    print("=" * 80)

    # 创建简单模型
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data = torch.randn(256, 128, device='cuda')

    # FP32 训练
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    time_fp32 = time.time() - start

    # 混合精度训练
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = output.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    time_amp = time.time() - start

    print(f"FP32 训练: {time_fp32:.4f}s (100次迭代)")
    print(f"混合精度训练: {time_amp:.4f}s (100次迭代)")
    print(f"加速比: {time_fp32/time_amp:.2f}x")

    if time_fp32 / time_amp > 1.3:
        print("\n✓ 强烈建议: 启用混合精度训练 (AMP)")
        print("  RTX 4090 的 Tensor Core 可以提供显著加速")
    print()


def check_current_bottleneck():
    """分析当前训练的瓶颈"""
    print("=" * 80)
    print("性能瓶颈分析")
    print("=" * 80)

    print("根据代码分析，发现以下瓶颈：")
    print()
    print("1. ❌ 数据传输瓶颈")
    print("   位置: Train.py:18")
    print("   问题: 每个batch都要调用 .cuda() 传输数据")
    print("   影响: 中等")
    print()

    print("2. ❌ 缺少 pin_memory")
    print("   位置: main.py:113")
    print("   问题: DataLoader 没有设置 pin_memory=True")
    print("   影响: 中等")
    print()

    print("3. ❌ CPU负采样慢")
    print("   位置: Dataset.py:159")
    print("   问题: random.sample 在CPU上执行，每个样本128次")
    print("   影响: 高 (这可能是主要瓶颈)")
    print()

    print("4. ❌ 没有混合精度训练")
    print("   位置: Train.py")
    print("   问题: 没有使用 torch.cuda.amp")
    print("   影响: 高 (4090的Tensor Core未被利用)")
    print()

    print("5. ❌ 不必要的 retain_graph")
    print("   位置: Train.py:19")
    print("   问题: backward(retain_graph=True) 保留计算图")
    print("   影响: 低")
    print()

    print("6. ⚠️  批量大小可能偏小")
    print("   当前: batch_size=256")
    print("   建议: 对于4090，可以尝试512或更大")
    print("   影响: 中等")
    print()


def provide_optimization_suggestions():
    """提供优化建议"""
    print("=" * 80)
    print("优化建议清单")
    print("=" * 80)

    suggestions = [
        {
            "priority": "高",
            "title": "启用混合精度训练",
            "benefit": "预计加速 1.5-2x",
            "code": """
# 在 Train.py 中添加
from torch.cuda.amp import autocast, GradScaler

def train(epoch, length, dataloader, model, optimizer, batch_size, writer):
    scaler = GradScaler()  # 添加这行

    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()

        # 使用混合精度
        with autocast():
            loss, model_loss, reg_loss = model.loss(
                user_tensor.cuda(), item_tensor.cuda()
            )

        scaler.scale(loss).backward()  # 修改这行
        scaler.step(optimizer)  # 修改这行
        scaler.update()  # 添加这行
"""
        },
        {
            "priority": "高",
            "title": "优化数据加载",
            "benefit": "预计加速 1.2-1.5x",
            "code": """
# 在 main.py 中修改 DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # 添加这行
    persistent_workers=True  # 添加这行 (PyTorch >= 1.7)
)
"""
        },
        {
            "priority": "高",
            "title": "移除不必要的 retain_graph",
            "benefit": "减少内存占用，略微加速",
            "code": """
# 在 Train.py:19 修改
loss.backward()  # 移除 retain_graph=True
"""
        },
        {
            "priority": "中",
            "title": "增大批量大小",
            "benefit": "预计加速 1.2-1.3x",
            "code": """
# 在运行命令中修改
python main.py --batch_size=512 ...  # 从256增加到512
# 或者更大，根据显存情况调整
"""
        },
        {
            "priority": "中",
            "title": "优化负采样（较复杂）",
            "benefit": "预计加速 1.3-2x",
            "code": """
# 方案1: 预先生成负样本
# 方案2: 使用GPU进行采样
# 方案3: 使用缓存采样策略
# (具体实现较复杂，需要重构 Dataset.py)
"""
        },
        {
            "priority": "低",
            "title": "启用 cuDNN benchmark",
            "benefit": "略微加速",
            "code": """
# 在 main.py 开头添加
torch.backends.cudnn.benchmark = True
"""
        }
    ]

    for i, sugg in enumerate(suggestions, 1):
        print(f"\n{i}. [{sugg['priority']}优先级] {sugg['title']}")
        print(f"   预期收益: {sugg['benefit']}")
        print(f"   代码修改:")
        for line in sugg['code'].strip().split('\n'):
            print(f"   {line}")

    print("\n" + "=" * 80)
    print("综合优化后，预计总加速: 2-4x")
    print("即速度可能从 ~3000 it/s 提升到 6000-12000 it/s")
    print("=" * 80)
    print()


def main():
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "CLCRec GPU 性能检查工具" + " " * 20 + "█")
    print("█" + " " * 25 + "For RTX 4090" + " " * 26 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    # 运行所有检查
    if not check_gpu_info():
        return

    check_cudnn()
    benchmark_data_transfer()
    benchmark_sampling()
    benchmark_mixed_precision()
    check_current_bottleneck()
    provide_optimization_suggestions()

    print("\n" + "=" * 80)
    print("检查完成！")
    print("=" * 80)
    print("\n要应用这些优化吗？我可以帮你修改代码。")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
