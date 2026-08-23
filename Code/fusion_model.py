import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """单模态作为Query，其他模态拼接作为Key/Value的交叉注意力"""
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query: [B, d_model]  单模态特征
        key_value: [B, L, d_model]  其他模态特征序列（L=模态数）
        """
        # 增加序列维度
        query = query.unsqueeze(1)  # [B, 1, d_model]
        attn_out, _ = self.attn(query, key_value, key_value)
        attn_out = attn_out.squeeze(1)  # [B, d_model]
        # 残差连接
        out = self.norm(query.squeeze(1) + self.dropout(attn_out))
        return out


class GatedFusion(nn.Module):
    """门控融合模块：为每个模态学习权重"""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, eeg, emg, pinch):
        """
        eeg, emg, pinch: [B, d_model]
        """
        concat = torch.cat([eeg, emg, pinch], dim=-1)  # [B, d_model*3]
        weights = self.gate(concat)  # [B, 3]
        # 加权求和
        fused = (weights[:, 0:1] * eeg +
                 weights[:, 1:2] * emg +
                 weights[:, 2:3] * pinch)
        return fused, weights


class MultiModalFusionModel(nn.Module):
    def __init__(self,
                 eeg_dim=12288,
                 emg_dim=64,
                 pinch_dim=128,
                 d_model=256,
                 num_classes=2,
                 dropout=0.3,
                 nhead=4):
        super().__init__()

        # 投影层：各模态降维到 d_model
        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model)
        )
        self.emg_proj = nn.Sequential(
            nn.Linear(emg_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model)
        )
        self.pinch_proj = nn.Sequential(
            nn.Linear(pinch_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model)
        )

        # 交叉注意力模块（每种模态分别作为Query）
        self.cross_attn_eeg = CrossModalAttention(d_model, nhead, dropout)
        self.cross_attn_emg = CrossModalAttention(d_model, nhead, dropout)
        self.cross_attn_pinch = CrossModalAttention(d_model, nhead, dropout)

        # 门控融合
        self.gated_fusion = GatedFusion(d_model)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg, emg, pinch):
        # 1. 投影
        eeg_feat = self.eeg_proj(eeg)      # [B, d_model]
        emg_feat = self.emg_proj(emg)      # [B, d_model]
        pinch_feat = self.pinch_proj(pinch)# [B, d_model]

        # 2. 构建交叉注意力的 Key/Value 序列
        # 对于每种模态，其他两种模态组成序列
        # EEG 作为 Query 时，Key/Value 为 EMG 和 Pinch 的堆叠
        kv_for_eeg = torch.stack([emg_feat, pinch_feat], dim=1)  # [B, 2, d_model]
        kv_for_emg = torch.stack([eeg_feat, pinch_feat], dim=1)
        kv_for_pinch = torch.stack([eeg_feat, emg_feat], dim=1)

        # 3. 交叉注意力增强
        eeg_enhanced = self.cross_attn_eeg(eeg_feat, kv_for_eeg)
        emg_enhanced = self.cross_attn_emg(emg_feat, kv_for_emg)
        pinch_enhanced = self.cross_attn_pinch(pinch_feat, kv_for_pinch)

        # 4. 门控融合
        fused, gate_weights = self.gated_fusion(eeg_enhanced, emg_enhanced, pinch_enhanced)

        # 5. 分类
        logits = self.classifier(fused)

        return logits, fused, gate_weights