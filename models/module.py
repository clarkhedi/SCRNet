import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientConceptRouter(nn.Module):
    """
    输入:
        img_feat: [B, L_i, D] 图像特征
        np_feat:  [B, L_n, D] 概念特征
        word_feat: [B, L_w, D] 文本特征

    输出:
        routed_feature: [B, D] 路由特征向量
    """
    def __init__(self, embed_dim, reduction_ratio=8, rank=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduced_dim = embed_dim // reduction_ratio

        # ====== SVD分解参数 ======
        # U: [D, r]，V: [d, r]，Sigma: [r]
        self.rank = rank  # SVD分解的秩 r
        self.U = nn.Parameter(torch.empty(embed_dim, rank))
        self.Sigma = nn.Parameter(torch.empty(rank))
        self.V = nn.Parameter(torch.empty(self.reduced_dim, rank))

        # SVD参数初始化
        self.reset_parameters()

        self.attn_transform = nn.Linear(self.reduced_dim, 3, bias=False)  # 生成注意力权重

        # 门控融合器
        self.gate = nn.Sequential(
            nn.Linear(self.reduced_dim * 3, self.reduced_dim),
            nn.ReLU(),
            nn.Linear(self.reduced_dim, 3),  # 三个模态的融合权重
            nn.Softmax(dim=1)
        )

        self.enhance_layer = self._create_enhance_block()  # 仅1层迭代

        self.output_proj = nn.Linear(self.reduced_dim, embed_dim) # 投影

    def reset_parameters(self):
        """
        用随机矩阵的SVD结果初始化 U, Σ, V
        """
        # 随机矩阵大小 D × d
        random_matrix = torch.randn(self.embed_dim, self.reduced_dim)

        # full_matrices=False 保证返回的形状是 (D, min(D,d)), (min(D,d),), (d, min(D,d))
        U, S, Vh = torch.linalg.svd(random_matrix, full_matrices=False)

        # 截断到 rank
        U = U[:, :self.rank]              # [D, r]
        S = S[:self.rank]                 # [r]
        V = Vh[:self.rank, :].T           # [d, r]

        # 复制到参数
        with torch.no_grad():
            self.U.copy_(U)
            self.Sigma.copy_(S)
            self.V.copy_(V)

    def _create_enhance_block(self):
        """创建轻量级特征增强块"""
        return nn.Sequential(
            nn.LayerNorm(self.reduced_dim),
            nn.Linear(self.reduced_dim, self.reduced_dim * 4),
            nn.GELU(),
            nn.Linear(self.reduced_dim * 4, self.reduced_dim)
        )

    def _masked_mean(self, feat, padding_mask):
        """掩码感知均值池化"""
        if padding_mask is None:
            return feat.mean(dim=1)  # [B, D]

        # 计算有效长度
        valid_mask = ~padding_mask
        valid_length = valid_mask.sum(dim=1, keepdim=True).float() + 1e-6  # 避免除以0

        # 应用掩码并求和
        masked_feat = feat * valid_mask.unsqueeze(-1)
        return masked_feat.sum(dim=1) / valid_length  # [B, D]

    def _svd_shared_proj(self, x):
        """
        使用 SVD 分解的低秩投影:
        y = x @ (U Σ V^T)
        x: [B, D]
        """
        # x @ U => [B, r]
        x_proj = x @ self.U  # [B, r]

        # 乘以 Σ（逐元素缩放）
        x_proj = x_proj * self.Sigma  # [B, r]

        # 再右乘 V^T => [B, d]
        return x_proj @ self.V.T    # [B, reduced_dim]

    def _masked_proj(self, feat, padding_mask):
        """带掩码的低维投影（低秩分解版）"""
        # 1. 掩码均值
        feat_mean = self._masked_mean(feat, padding_mask)  # [B, D]
        # 2. 低秩共享投影
        return self._svd_shared_proj(feat_mean)         # [B, reduced_dim]

    def forward(self, img_feat, np_feat, word_feat,
                np_key_padding_mask=None, word_key_padding_mask=None):
        img_proj = self._svd_shared_proj(img_feat.mean(1))   # [B, reduced_dim]
        np_proj = self._masked_proj(np_feat, np_key_padding_mask)    # [B, reduced_dim]
        word_proj = self._masked_proj(word_feat, word_key_padding_mask) # [B, reduced_dim]

        all_features = torch.stack([img_proj, np_proj, word_proj], dim=1)  # [B, 3, reduced_dim]
        attn_map = self.attn_transform(all_features).softmax(dim=-1)  # [B, 3, 3]

        enhanced_features = torch.matmul(attn_map, all_features)  # [B, 3, reduced_dim]
        img_enhanced, np_enhanced, word_enhanced = torch.unbind(enhanced_features, dim=1)

        features_concat = torch.cat([img_enhanced, np_enhanced, word_enhanced], dim=-1)  # [B, reduced_dim*3]
        weights = self.gate(features_concat)  # [B, 3]

        # 加权融合
        fused_feature = (
            weights[:, 0].unsqueeze(1) * img_enhanced +
            weights[:, 1].unsqueeze(1) * np_enhanced +
            weights[:, 2].unsqueeze(1) * word_enhanced
        )  # [B, reduced_dim]

        routed = fused_feature + self.enhance_layer(fused_feature)  # 残差连接
        return self.output_proj(routed)


class ConceptExpertSystem(nn.Module):
    def __init__(self, embed_dim, num_concepts, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.num_concepts = num_concepts

        # 专家池：每个专家专注于不同类型的概念
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, num_concepts),
                nn.Sigmoid()
            ) for _ in range(num_experts)
        ])

        # 门控网络：动态选择专家组合
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, img_feat, text_feat):
        """
        img_feat: [B, D] 全局图像特征
        text_feat: [B, D] 全局文本特征
        返回: [B, num_concepts] 概念权重
        """
        fused_feat = torch.cat([img_feat, text_feat], dim=1)
        gate_weights = self.gate_network(fused_feat)  # [B, num_experts]

        expert_outputs = torch.stack([expert(fused_feat) for expert in self.experts], dim=1)  # [B, num_experts, C]

        # 加权组合
        concept_weights = torch.einsum('be,bec->bc', gate_weights, expert_outputs)  # [B, C]

        # 归一化
        concept_weights = concept_weights / concept_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return concept_weights


class EnhancedConceptMatcher(nn.Module):
    def __init__(self, embed_dim, concept_dim, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts

        self.expert_system = ConceptExpertSystem(embed_dim, num_concepts)

        # 概念匹配分类器
        self.itm_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(concept_dim, concept_dim//2),
                nn.ReLU(),
                nn.Linear(concept_dim//2, 2)
            ) for _ in range(num_concepts)
        ])

        # 概念一致性约束
        self.consistency_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, img_feat, text_feat, concepts_pos, concepts_neg,
               concept_pos_atts, concept_neg_atts, itm_labels):
        concept_weights = self.expert_system(img_feat, text_feat) # 生成专家权重

        loss_cm = 0.
        valid_concepts = 0

        for i in range(self.num_concepts):
            cm_embeddings = torch.cat([concepts_pos[:, i], concepts_neg[:, i]], dim=0)
            cm_concept_atts = torch.cat([concept_pos_atts[:, i], concept_neg_atts[:, i]], dim=0)

            # 仅处理有效概念
            valid_mask = cm_concept_atts != 0
            if not valid_mask.any():
                continue

            cm_output = self.itm_heads[i](cm_embeddings)

            # 计算当前概念损失
            concept_loss = F.cross_entropy(
                cm_output[valid_mask],
                itm_labels[valid_mask],
                reduction='none'
            )

            # 专家权重加权
            batch_weights = concept_weights[:, i][valid_mask]

            loss_cm += (concept_loss * batch_weights).mean()
            valid_concepts += 1

        if valid_concepts > 0:
            loss_cm = loss_cm / valid_concepts

        # 概念一致性约束
        mean_weights = concept_weights.mean(dim=0)
        uniform_prior = torch.ones_like(mean_weights) / self.num_concepts
        loss_consistency = self.consistency_loss(
            F.log_softmax(mean_weights, dim=-1),
            uniform_prior
        )

        return loss_cm + loss_consistency, concept_weights
