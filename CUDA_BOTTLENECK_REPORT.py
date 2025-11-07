#!/usr/bin/env python3
"""
检查CUDA计算瓶颈
"""

print("=" * 80)
print("发现的CUDA计算问题")
print("=" * 80)

print("""
通过代码审查，发现以下可能导致GPU利用率低的问题：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题1: 初始化时的CPU tensor ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

位置: model_CLCRec.py:98-100

当前代码:
    self.contrastive_loss_1 = torch.tensor(0.0)
    self.contrastive_loss_2 = torch.tensor(0.0)
    self.neighbor_item_loss = torch.tensor(0.0)

问题:
- torch.tensor(0.0) 默认在CPU上创建
- 虽然后续会被GPU tensor覆盖，但初始化不规范

影响: 低（因为会被覆盖）

修复:
    self.contrastive_loss_1 = torch.tensor(0.0, device='cuda')
    self.contrastive_loss_2 = torch.tensor(0.0, device='cuda')
    self.neighbor_item_loss = torch.tensor(0.0, device='cuda')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题2: torch.randint 没有指定device ⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

位置: model_CLCRec.py:291

当前代码:
    rand_index = torch.randint(0, all_item_embedding.size(0), (num_to_replace,)).cuda()

问题:
- torch.randint 默认在CPU上生成，然后.cuda()移到GPU
- 这会产生CPU->GPU数据传输
- 每个batch都会执行一次

影响: 中等

修复:
    rand_index = torch.randint(
        0, all_item_embedding.size(0), (num_to_replace,),
        device=all_item_embedding.device
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题3: encoder中不必要的.cuda()调用 ⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

位置: model_CLCRec.py:151

当前代码:
    t_feat = F.normalize(
        scatter_mean_manual(
            self.t_feat[self.word_tensor[1]],
            self.word_tensor[0],
            dim=0
        )
    ).cuda()

问题:
- 模型已经在GPU上，scatter_mean_manual的输出应该已经在GPU上
- 显式的.cuda()调用可能暗示之前的操作在CPU上
- 需要检查scatter_mean_manual实现

影响: 中等（如果scatter_mean在CPU上则影响很大）

修复:
- 检查scatter_mean_manual是否在GPU上执行
- 移除不必要的.cuda()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题4: 邻居聚合中的循环操作 ⭐⭐⭐⭐⭐ (可能是主要瓶颈!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

位置: model_CLCRec.py:175-220

当前代码:
    for i, user in enumerate(users):
        user_id = int(user.item())  # ⚠️ GPU->CPU同步！
        neighbors = self.user_neighbors.get(user_id, [])
        ...

问题:
- user.item() 会导致GPU->CPU同步，非常慢！
- 每个batch中的每个用户都要执行一次
- 循环内有大量Python操作
- 这可能是GPU利用率低的最大原因！

影响: 非常高！

每个batch的开销:
- batch_size=256，每个batch要同步256次
- 每次同步约0.1-0.5ms
- 总开销: 25-128ms/batch
- 这会严重拖慢训练速度！

修复建议:
1. 预计算邻居信息到tensor
2. 使用向量化操作替代循环
3. 或者禁用邻居损失（use_neighbor_loss=False）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题5: 数据加载时的张量创建 ⭐⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

位置: Dataset.py:162-163

当前代码:
    user_tensor = torch.LongTensor([user] * (self.num_neg + 1))
    item_tensor = torch.LongTensor([pos_item] + neg_item)

问题:
- 在CPU上创建tensor
- 在Train.py:26中才.cuda()
- 每个样本都要CPU->GPU传输

影响: 高

这个已经通过pin_memory=True部分缓解，但仍有优化空间

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("=" * 80)
print("修复优先级")
print("=" * 80)

print("""
高优先级（严重影响性能）：

1. ⭐⭐⭐⭐⭐ 禁用或优化邻居聚合
   - user.item() 导致频繁的GPU->CPU同步
   - 建议：先禁用邻居损失测试速度

2. ⭐⭐⭐⭐ 增大batch_size
   - 当前256太小，无法充分利用GPU
   - 建议：1024或2048

中优先级（有一定影响）：

3. ⭐⭐⭐ 优化randint生成
   - 避免CPU->GPU传输

4. ⭐⭐ 检查scatter_mean_manual实现

低优先级（影响较小）：

5. ⭐ 修复初始化的device
""")

print("=" * 80)
print("快速测试方案")
print("=" * 80)

print("""
测试1: 禁用邻居损失（最简单，预期效果最好）
-----------------------------------------------

修改 main.py:126-127:

# 原来
model = CLCRec(num_user, num_item, num_warm_item, train_data, reg_weight, dim_E,
               v_feat, a_feat, t_feat, temp_value, num_neg, lr_lambda, is_word, num_sample).cuda()

# 改为（添加 use_neighbor_loss=False）
model = CLCRec(num_user, num_item, num_warm_item, train_data, reg_weight, dim_E,
               v_feat, a_feat, t_feat, temp_value, num_neg, lr_lambda, is_word, num_sample,
               use_neighbor_loss=False).cuda()

预期效果:
- 消除GPU->CPU同步瓶颈
- GPU利用率: +20-30%
- 速度: +30-50%

测试2: 组合优化
-----------------------------------------------

禁用邻居损失 + 增大batch

python main.py \\
  --batch_size=1024 \\
  --num_workers=8 \\
  --l_r=0.001 --reg_weight=0.1 --num_neg=128 \\
  --has_a=True --has_t=True --has_v=True \\
  --lr_lambda=0.5 --temp_value=2.0 --num_sample=0.5

预期效果:
- GPU利用率: 90-100%
- 速度: 3400 -> 8000-12000 it/s (2.5-3.5x)
""")

print("=" * 80)
print("诊断命令")
print("=" * 80)

print("""
1. 监控GPU利用率:
   watch -n 1 nvidia-smi

2. 检查是否有CPU->GPU同步:
   使用PyTorch Profiler:

   from torch.profiler import profile, ProfilerActivity

   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       # 训练一个batch
       ...

   print(prof.key_averages().table(sort_by="cuda_time_total"))

   查找 "cudaStreamSynchronize" 或 "cudaMemcpy" 调用
""")
