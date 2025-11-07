from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


##########################################################################
# 改进的 scatter mean 实现（使用 PyTorch 原生操作，避免循环）
##########################################################################

def scatter_mean_manual(src, index, dim=0):
    """
    高效的 scatter mean 实现，替代 torch_scatter.scatter

    Args:
        src: 源张量 [N, D]
        index: 索引张量 [N]
        dim: 聚合维度

    Returns:
        out: 聚合后的张量 [max(index)+1, D]
    """
    if src.size(0) == 0:
        return src

    num_nodes = index.max().item() + 1
    out = torch.zeros(num_nodes, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(num_nodes, 1, dtype=torch.float, device=src.device)

    # 使用 scatter_add_ 进行累加（比循环快得多）
    out.scatter_add_(0, index.unsqueeze(1).expand(-1, src.size(1)), src)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones_like(index, dtype=torch.float).unsqueeze(1))

    # 计算平均值，避免除零
    count = count.clamp(min=1)
    out = out / count

    return out


##########################################################################

class CLCRec(torch.nn.Module):
    def __init__(self, num_user, num_item, num_warm_item, edge_index, reg_weight, dim_E, v_feat, a_feat, t_feat,
                 temp_value, num_neg, lr_lambda, is_word, num_sample=0.5, use_neighbor_loss=False):
        super(CLCRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_warm_item = num_warm_item
        self.num_neg = num_neg
        self.lr_lambda = lr_lambda
        self.reg_weight = reg_weight
        self.temp_value = temp_value
        self.dim_E = dim_E
        self.is_word = is_word
        self.num_sample = num_sample
        self.use_neighbor_loss = use_neighbor_loss  # 🔧 新增：控制是否使用邻居损失

        # ID嵌入
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))
        self.dim_feat = 0

        # 多模态特征
        if v_feat is not None:
            self.v_feat = F.normalize(v_feat, dim=1)
            self.dim_feat += self.v_feat.size(1)
        else:
            self.v_feat = None

        if a_feat is not None:
            self.a_feat = F.normalize(a_feat, dim=1)
            self.dim_feat += self.a_feat.size(1)
        else:
            self.a_feat = None

        if t_feat is not None:
            if is_word:
                self.t_feat = nn.Parameter(nn.init.xavier_normal_(torch.rand((torch.max(t_feat[1]).item() + 1, 128))))
                self.word_tensor = t_feat
            else:
                self.t_feat = F.normalize(t_feat, dim=1)
            self.dim_feat += self.t_feat.size(1) if not is_word else 128
        else:
            self.t_feat = None

        # 编码器
        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, dim_E)

        # 用户-物品交互图
        self.build_user_item_graph(edge_index)

        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))).cuda()

        # 初始化损失变量
        # 🔧 修复：在GPU上创建tensor
        self.contrastive_loss_1 = torch.tensor(0.0, device='cuda')
        self.contrastive_loss_2 = torch.tensor(0.0, device='cuda')
        self.neighbor_item_loss = torch.tensor(0.0, device='cuda')

    def build_user_item_graph(self, train_data):
        """构建用户-物品交互图，用于查找邻居"""
        # 🚀 性能优化：如果禁用邻居损失，跳过图构建以节省初始化时间
        if not self.use_neighbor_loss:
            print("⚠️  邻居损失已禁用，跳过用户-物品图构建（加速初始化）")
            self.user_items = {}
            self.item_users = {}
            self.user_neighbors = {}
            return

        self.user_items = {}  # 用户交互的物品
        self.item_users = {}  # 物品被哪些用户交互

        for user, item in train_data:
            # 🔧 修复：转换为Python int
            user = int(user)
            item = int(item)

            if user not in self.user_items:
                self.user_items[user] = set()
            self.user_items[user].add(item)

            if item not in self.item_users:
                self.item_users[item] = set()
            self.item_users[item].add(user)

        # 预计算用户邻居（有共同物品的用户）
        print("Building user neighbor graph...")
        self.user_neighbors = {}
        for user in tqdm(range(self.num_user)):
            neighbors = set()
            if user in self.user_items:
                for item in self.user_items[user]:
                    if item in self.item_users:
                        neighbors.update(self.item_users[item])
                neighbors.discard(user)  # 移除自己
            self.user_neighbors[user] = list(neighbors)[:50]  # 最多50个邻居

    def encoder(self):
        """编码多模态特征"""
        feature_list = []

        if self.v_feat is not None:
            feature_list.append(self.v_feat)

        if self.a_feat is not None:
            feature_list.append(self.a_feat)

        if self.t_feat is not None:
            if self.is_word:
                # 使用改进的 scatter_mean
                # 🔧 移除不必要的.cuda()，scatter_mean输出已在GPU上
                t_feat = F.normalize(
                    scatter_mean_manual(
                        self.t_feat[self.word_tensor[1]],
                        self.word_tensor[0],
                        dim=0
                    )
                )
                feature_list.append(t_feat)
            else:
                feature_list.append(self.t_feat)

        # 拼接特征
        if len(feature_list) == 0:
            # 如果没有任何特征，返回零向量
            feature = torch.zeros(self.num_item, self.dim_E).cuda()
        else:
            feature = torch.cat(feature_list, dim=1)
            feature = F.leaky_relu(self.encoder_layer1(feature))
            feature = self.encoder_layer2(feature)

        return feature

    def get_neighbor_aggregation(self, users):
        """获取用户邻居的聚合特征和共同物品的聚合特征"""
        batch_size = users.size(0)
        neighbor_embeds = []
        common_item_embeds = []

        feature = self.encoder()

        for i, user in enumerate(users):
            # 🔧 修复：确保user_id是Python int
            user_id = int(user.item())

            # 获取邻居用户
            neighbors = self.user_neighbors.get(user_id, [])

            if len(neighbors) > 0:
                # 聚合邻居用户embedding (取top-k个)
                k = min(10, len(neighbors))
                neighbor_ids = neighbors[:k]
                neighbor_embed = self.id_embedding[neighbor_ids].mean(dim=0)

                # 找共同物品
                common_items = set()
                for neighbor in neighbor_ids:
                    neighbor = int(neighbor)  # 🔧 确保是int
                    if neighbor in self.user_items:
                        common_items.update(self.user_items[neighbor])

                # 过滤掉用户自己交互过的物品
                if user_id in self.user_items:
                    common_items -= self.user_items[user_id]

                common_items = list(common_items)[:20]  # 最多20个共同物品

                if len(common_items) > 0:
                    # 聚合共同物品的特征 - 修复索引问题
                    item_indices = []
                    for item in common_items:
                        item = int(item)  # 🔧 确保是int
                        idx = item - self.num_user
                        # 确保索引在有效范围内
                        if 0 <= idx < feature.size(0):
                            item_indices.append(idx)

                    if len(item_indices) > 0:
                        common_item_embed = feature[item_indices].mean(dim=0)
                    else:
                        common_item_embed = neighbor_embed.clone()
                else:
                    common_item_embed = neighbor_embed.clone()
            else:
                # 没有邻居，使用用户自己的embedding
                neighbor_embed = self.id_embedding[user_id]
                common_item_embed = neighbor_embed.clone()

            neighbor_embeds.append(neighbor_embed)
            common_item_embeds.append(common_item_embed)

        neighbor_embeds = torch.stack(neighbor_embeds)
        common_item_embeds = torch.stack(common_item_embeds)

        return neighbor_embeds, common_item_embeds

    def loss_contrastive(self, tensor_anchor, tensor_all, temp_value):
        """原始对比损失"""
        all_score = torch.exp(torch.sum(tensor_anchor * tensor_all, dim=1) / temp_value).view(-1, 1 + self.num_neg)
        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)

        # 添加数值稳定性
        contrastive_loss = (-torch.log(pos_score / (all_score + 1e-8) + 1e-8)).mean()
        return contrastive_loss

    def loss_neighbor_item(self, neighbor_embed, item_embed, temp_value):
        """新增：用户邻居 vs 共同物品的对比损失"""
        neighbor_embed = F.normalize(neighbor_embed, dim=1)
        item_embed = F.normalize(item_embed, dim=1)

        # 计算相似度
        pos_score = torch.exp(torch.sum(neighbor_embed * item_embed, dim=1) / temp_value)

        # 负样本：batch内其他样本
        neg_score = torch.exp(torch.matmul(neighbor_embed, item_embed.t()) / temp_value)
        neg_score = torch.sum(neg_score, dim=1) - pos_score  # 排除自己

        loss = -torch.log(pos_score / (pos_score + neg_score + 1e-8)).mean()
        return loss

    def forward(self, user_tensor, item_tensor):
        # 处理张量形状
        pos_item_tensor = item_tensor[:, 0].unsqueeze(1)
        pos_item_tensor = pos_item_tensor.repeat(1, 1 + self.num_neg).view(-1, 1).squeeze()

        user_tensor_flat = user_tensor.view(-1, 1).squeeze()
        item_tensor_flat = item_tensor.view(-1, 1).squeeze()

        # 获取唯一用户（用于邻居聚合）
        unique_users = user_tensor[:, 0]

        # 编码特征
        feature = self.encoder()

        # 修复：确保item索引在有效范围内
        item_indices = item_tensor_flat - self.num_user
        valid_mask = (item_indices >= 0) & (item_indices < feature.size(0))

        if not valid_mask.all():
            print(f"Warning: Some item indices out of range. Valid: {valid_mask.sum()}/{len(valid_mask)}")
            item_indices = torch.clamp(item_indices, 0, feature.size(0) - 1)

        all_item_feat = feature[item_indices]

        # Embeddings
        user_embedding = self.id_embedding[user_tensor_flat]
        pos_item_embedding = self.id_embedding[pos_item_tensor]
        all_item_embedding = self.id_embedding[item_tensor_flat]

        # 原始对比损失
        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        num_to_replace = int(all_item_embedding.size(0) * self.num_sample)
        if num_to_replace > 0:
            # 🔧 修复：直接在GPU上生成随机索引，避免CPU->GPU传输
            rand_index = torch.randint(
                0, all_item_embedding.size(0), (num_to_replace,),
                device=all_item_embedding.device
            )
            # 🔧 修复混合精度训练的类型不匹配问题
            all_item_input[rand_index] = all_item_feat[rand_index].to(all_item_input.dtype)

        self.contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temp_value)
        self.contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        # 新增：邻居-物品对比损失（可选）
        if self.use_neighbor_loss:
            try:
                neighbor_embeds, common_item_embeds = self.get_neighbor_aggregation(unique_users)
                self.neighbor_item_loss = self.loss_neighbor_item(neighbor_embeds, common_item_embeds, self.temp_value)
            except Exception as e:
                print(f"Warning: Neighbor aggregation failed: {e}")
                self.neighbor_item_loss = torch.tensor(0.0).cuda()
        else:
            # 禁用邻居损失以提升训练速度
            self.neighbor_item_loss = torch.tensor(0.0).cuda()

        # 正则化
        reg_loss = ((torch.sqrt((user_embedding ** 2).sum(1))).mean() +
                    (torch.sqrt((all_item_embedding ** 2).sum(1))).mean()) / 2

        # 更新result
        # 🔧 确保类型一致以支持混合精度训练
        warm_embeddings = self.id_embedding[:self.num_user + self.num_warm_item]
        cold_features = feature[self.num_warm_item:].to(warm_embeddings.dtype)
        self.result = torch.cat((warm_embeddings, cold_features), dim=0)

        # 总损失：原始损失 + 新的邻居-物品损失
        total_loss = (self.contrastive_loss_1 * self.lr_lambda +
                      self.contrastive_loss_2 * (1 - self.lr_lambda) +
                      self.neighbor_item_loss * 0.1)  # 0.1是新损失的权重

        return total_loss, reg_loss

    def loss(self, user_tensor, item_tensor):
        contrastive_loss, reg_loss = self.forward(user_tensor, item_tensor)
        reg_loss = self.reg_weight * reg_loss
        return reg_loss + contrastive_loss, self.contrastive_loss_2 + reg_loss, reg_loss