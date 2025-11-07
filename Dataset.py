import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def data_load(dataset, has_v=True, has_a=True, has_t=True):
    dir_str = './Data/' + dataset

    # 加载交互数据
    train_data = np.load(dir_str + '/train.npy', allow_pickle=True)
    val_data = np.load(dir_str + '/val_full.npy', allow_pickle=True)
    val_warm_data = np.load(dir_str + '/val_warm.npy', allow_pickle=True)
    val_cold_data = np.load(dir_str + '/val_cold.npy', allow_pickle=True)
    test_data = np.load(dir_str + '/test_full.npy', allow_pickle=True)
    test_warm_data = np.load(dir_str + '/test_warm.npy', allow_pickle=True)
    test_cold_data = np.load(dir_str + '/test_cold.npy', allow_pickle=True)

    if dataset == 'movielens':
        num_user = 55485
        num_item = 5986
        num_warm_item = 5119

        # 检查特征文件是否存在
        v_feat_path = dir_str + '/feat_v.npy'
        a_feat_path = dir_str + '/feat_a.npy'
        t_feat_path = dir_str + '/feat_t.npy'

        # 加载视觉特征
        if has_v and os.path.exists(v_feat_path):
            v_feat = torch.tensor(np.load(v_feat_path, allow_pickle=True), dtype=torch.float).cuda()
            print(f"✓ 加载视觉特征: {v_feat.shape}")
        else:
            v_feat = None
            if has_v:
                print(f"⚠ 视觉特征文件不存在，已忽略")

        # 加载音频特征
        if has_a and os.path.exists(a_feat_path):
            a_feat = torch.tensor(np.load(a_feat_path, allow_pickle=True), dtype=torch.float).cuda()
            print(f"✓ 加载音频特征: {a_feat.shape}")
        else:
            a_feat = None
            if has_a:
                print(f"⚠ 音频特征文件不存在，已忽略")

        # 加载文本特征
        if has_t and os.path.exists(t_feat_path):
            t_feat = torch.tensor(np.load(t_feat_path, allow_pickle=True), dtype=torch.float).cuda()
            print(f"✓ 加载文本特征: {t_feat.shape}")
        else:
            t_feat = None
            if has_t:
                print(f"⚠ 文本特征文件不存在，已忽略")

    elif dataset == 'amazon':
        num_user = 27044
        num_item = 86506
        num_warm_item = 68810

        v_feat_path = dir_str + '/feat_v.pt'
        if os.path.exists(v_feat_path):
            v_feat = torch.load(v_feat_path)
            print(f"✓ 加载视觉特征: {v_feat.shape}")
        else:
            v_feat = None
            print(f"⚠ 视觉特征文件不存在，已忽略")

        a_feat = None
        t_feat = None

    elif dataset == 'tiktok':
        num_user = 32309
        num_item = 57832 + 8624
        num_warm_item = 57832

        v_feat_path = dir_str + '/feat_v.pt'
        a_feat_path = dir_str + '/feat_a.pt'
        t_feat_path = dir_str + '/feat_t.pt'

        if has_v and os.path.exists(v_feat_path):
            v_feat = torch.load(v_feat_path)
            v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
            print(f"✓ 加载视觉特征: {v_feat.shape}")
        else:
            v_feat = None
            if has_v:
                print(f"⚠ 视觉特征文件不存在，已忽略")

        if has_a and os.path.exists(a_feat_path):
            a_feat = torch.load(a_feat_path)
            a_feat = torch.tensor(a_feat, dtype=torch.float).cuda()
            print(f"✓ 加载音频特征: {a_feat.shape}")
        else:
            a_feat = None
            if has_a:
                print(f"⚠ 音频特征文件不存在，已忽略")

        if os.path.exists(t_feat_path):
            t_feat = torch.load(t_feat_path).cuda()
            print(f"✓ 加载文本特征")
        else:
            # Tiktok的文本特征是词索引，这里简化处理
            t_feat = None
            print(f"⚠ 文本特征文件不存在，禁用文本特征")

    elif dataset == 'kwai':
        num_user = 7010
        num_item = 86483
        num_warm_item = 74470

        v_feat_path = dir_str + '/feat_v.npy'
        if os.path.exists(v_feat_path):
            v_feat = np.load(v_feat_path)
            v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
            print(f"✓ 加载视觉特征: {v_feat.shape}")
        else:
            v_feat = None
            print(f"⚠ 视觉特征文件不存在，已忽略")

        a_feat = t_feat = None

    return num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, dataset, train_data, num_neg):
        self.train_data = train_data
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict

        cold_set_path = './Data/' + dataset + '/cold_set.npy'
        if os.path.exists(cold_set_path):
            self.cold_set = set(np.load(cold_set_path))
        else:
            print(f"⚠ cold_set.npy 不存在，假设所有物品都是warm")
            self.cold_set = set()

        self.all_set = set(range(num_user, num_user + num_item)) - self.cold_set

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item = self.train_data[index]

        # 🔧 修复：将numpy类型转换为Python int
        user = int(user)
        pos_item = int(pos_item)

        # 负采样
        neg_item = random.sample(self.all_set - set(self.user_item_dict[user]), self.num_neg)

        # 创建张量
        user_tensor = torch.LongTensor([user] * (self.num_neg + 1))
        item_tensor = torch.LongTensor([pos_item] + neg_item)

        return user_tensor, item_tensor