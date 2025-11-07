"""
优化版数据集 - 预生成负样本以加速训练

主要优化：
1. 预生成所有负样本，避免每次 __getitem__ 时进行CPU采样
2. 支持定期重新采样以保持随机性
3. 显著减少集合运算开销
"""

import time
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TrainingDatasetOptimized(Dataset):
    """优化版训练数据集 - 预生成负样本"""

    def __init__(self, num_user, num_item, user_item_dict, dataset, train_data, num_neg,
                 resample_epochs=5):
        """
        Args:
            resample_epochs: 每隔多少个epoch重新生成负样本（默认5）
                            设为0表示只生成一次
        """
        self.train_data = train_data
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.resample_epochs = resample_epochs
        self.current_epoch = 0

        cold_set_path = './Data/' + dataset + '/cold_set.npy'
        if os.path.exists(cold_set_path):
            self.cold_set = set(np.load(cold_set_path))
        else:
            print(f"⚠ cold_set.npy 不存在，假设所有物品都是warm")
            self.cold_set = set()

        self.all_set = set(range(num_user, num_user + num_item)) - self.cold_set

        # 🚀 预生成负样本
        print("\n🚀 预生成负样本以加速训练...")
        self.negative_samples = None
        self.generate_negative_samples()

    def generate_negative_samples(self):
        """预生成所有负样本"""
        start_time = time.time()

        self.negative_samples = []

        print(f"   生成 {len(self.train_data)} 个样本的负样本 (num_neg={self.num_neg})...")

        for user, pos_item in tqdm(self.train_data, desc="   Generating negatives", ncols=80):
            user = int(user)
            # 候选负样本集合
            candidate_set = self.all_set - set(self.user_item_dict[user])
            # 采样
            neg_items = random.sample(candidate_set, self.num_neg)
            self.negative_samples.append(neg_items)

        elapsed = time.time() - start_time
        print(f"✓ 负样本生成完成！耗时: {elapsed:.2f}s")
        print(f"   这个开销在整个训练中只需要付出一次（或每{self.resample_epochs}个epoch一次）\n")

    def resample_if_needed(self, epoch):
        """根据需要重新生成负样本"""
        if self.resample_epochs > 0 and epoch % self.resample_epochs == 0 and epoch > 0:
            print(f"\n>>> Epoch {epoch}: 重新生成负样本...")
            self.generate_negative_samples()
        self.current_epoch = epoch

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item = self.train_data[index]

        # 🔧 修复：将numpy类型转换为Python int
        user = int(user)
        pos_item = int(pos_item)

        # 🚀 直接使用预生成的负样本（超快！）
        neg_items = self.negative_samples[index]

        # 创建张量
        user_tensor = torch.LongTensor([user] * (self.num_neg + 1))
        item_tensor = torch.LongTensor([pos_item] + neg_items)

        return user_tensor, item_tensor


class TrainingDatasetGPU(Dataset):
    """GPU版训练数据集 - 在GPU上采样（实验性）"""

    def __init__(self, num_user, num_item, user_item_dict, dataset, train_data, num_neg):
        """
        实验性版本：尝试在GPU上进行负采样
        注意：可能不稳定，需要更多测试
        """
        self.train_data = train_data
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict

        cold_set_path = './Data/' + dataset + '/cold_set.npy'
        if os.path.exists(cold_set_path):
            self.cold_set = set(np.load(cold_set_path))
        else:
            self.cold_set = set()

        self.all_items = torch.tensor(
            list(set(range(num_user, num_user + num_item)) - self.cold_set),
            dtype=torch.long,
            device='cuda'
        )

        # 预计算每个用户的正样本mask
        print("\n🚀 预计算用户-物品mask...")
        self.user_pos_masks = {}
        for user in tqdm(range(num_user), desc="   Building masks", ncols=80):
            if user in user_item_dict:
                pos_items = torch.tensor(
                    list(user_item_dict[user]),
                    dtype=torch.long,
                    device='cuda'
                )
                # 创建mask
                mask = torch.ones(len(self.all_items), dtype=torch.bool, device='cuda')
                for item in pos_items:
                    item_idx = (self.all_items == item).nonzero(as_tuple=True)[0]
                    if len(item_idx) > 0:
                        mask[item_idx[0]] = False
                self.user_pos_masks[user] = mask

        print("✓ Mask预计算完成！\n")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item = self.train_data[index]
        user = int(user)
        pos_item = int(pos_item)

        # 🚀 在GPU上采样
        if user in self.user_pos_masks:
            mask = self.user_pos_masks[user]
            candidate_items = self.all_items[mask]

            # GPU上随机采样
            perm = torch.randperm(len(candidate_items), device='cuda')[:self.num_neg]
            neg_items = candidate_items[perm].cpu().tolist()
        else:
            # fallback to CPU
            neg_items = random.sample(
                list(set(range(self.num_user, self.num_user + self.num_item)) - self.cold_set),
                self.num_neg
            )

        user_tensor = torch.LongTensor([user] * (self.num_neg + 1))
        item_tensor = torch.LongTensor([pos_item] + neg_items)

        return user_tensor, item_tensor


# 保持原始版本以便兼容
from Dataset import TrainingDataset as TrainingDatasetOriginal


def benchmark_datasets(num_samples=1000):
    """对比测试不同数据集实现的速度"""
    print("\n" + "=" * 80)
    print("数据集性能对比测试")
    print("=" * 80)

    # 这里需要实际的数据来测试
    # 仅作为示例代码
    print("\n运行方法：")
    print("1. 在 main.py 中导入: from Dataset_optimized import TrainingDatasetOptimized")
    print("2. 替换数据集创建代码")
    print("3. 观察训练速度提升")


if __name__ == '__main__':
    benchmark_datasets()
