import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def train(epoch, length, dataloader, model, optimizer, batch_size, writer):
    model.train()
    print('Now, training start ...')

    # 🚀 初始化混合精度训练
    scaler = GradScaler()

    sum_loss = 0.0
    sum_model_loss = 0.0
    sum_reg_loss = 0.0
    sum_neighbor_loss = 0.0  # 新增：邻居-物品损失
    step = 0.0
    pbar = tqdm(total=length)

    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()

        # 🚀 使用混合精度前向传播
        with autocast():
            loss, model_loss, reg_loss = model.loss(user_tensor.cuda(), item_tensor.cuda())

        # 🚀 使用 GradScaler 进行反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_loss += loss.cpu().item()
        sum_model_loss += model_loss.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()
        if hasattr(model, 'neighbor_item_loss'):
            sum_neighbor_loss += model.neighbor_item_loss.cpu().item()

        pbar.update(batch_size)
        step += 1.0

    pbar.close()

    print('----------------- Loss Summary -----------------')
    print(f'Total Loss: {sum_loss / step:.4f}')
    print(f'Model Loss: {sum_model_loss / step:.4f}')
    print(f'Reg Loss: {sum_reg_loss / step:.4f}')
    print(f'Neighbor-Item Loss: {sum_neighbor_loss / step:.4f}')
    print('-----------------------------------------------')

    return loss, sum_neighbor_loss / step