"""
baseline_models.py

主流多模态基线模型。适用于当前 FinalFusion 工程中已经提取好的三模态特征：
    EEG:   [N, eeg_dim]
    sEMG:  [N, emg_dim]
    Pinch: [N, pinch_dim]

这些模型都采用相同的输入形式 forward(eeg, emg, pinch)，返回 logits。
建议和 fusion_model.py 放在同一个目录下。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalProjector(nn.Module):
    """将 EEG、EMG、Pinch 三个异构特征投影到统一维度 d_model。"""

    def __init__(self, eeg_dim: int, emg_dim: int, pinch_dim: int, d_model: int = 256, dropout: float = 0.3):
        super().__init__()
        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_dim, max(512, d_model)),
            nn.BatchNorm1d(max(512, d_model)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(512, d_model), d_model),
        )
        self.emg_proj = nn.Sequential(
            nn.Linear(emg_dim, max(128, d_model // 2)),
            nn.BatchNorm1d(max(128, d_model // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(128, d_model // 2), d_model),
        )
        self.pinch_proj = nn.Sequential(
            nn.Linear(pinch_dim, max(128, d_model // 2)),
            nn.BatchNorm1d(max(128, d_model // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(128, d_model // 2), d_model),
        )

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor) -> torch.Tensor:
        eeg_feat = self.eeg_proj(eeg)
        emg_feat = self.emg_proj(emg)
        pinch_feat = self.pinch_proj(pinch)
        # [B, 3, d_model]，三个 token 分别代表 EEG、EMG、Pinch
        return torch.stack([eeg_feat, emg_feat, pinch_feat], dim=1)


class MultimodalTransformerEncoder(nn.Module):
    """
    Baseline 1：Multimodal Transformer Encoder。

    设计目的：
    - 用普通 Transformer 自注意力建模三模态 token 之间的关系；
    - 与本文模型对比时，可以说明普通 self-attention 与本文 cross-attention + gating 的差异。
    """

    def __init__(
        self,
        eeg_dim: int,
        emg_dim: int,
        pinch_dim: int,
        d_model: int = 256,
        num_classes: int = 2,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projector = ModalProjector(eeg_dim, emg_dim, pinch_dim, d_model, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 4, d_model))  # CLS + 3 modalities

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor) -> torch.Tensor:
        tokens = self.projector(eeg, emg, pinch)  # [B, 3, d]
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.modality_embedding
        x = self.encoder(x)
        pooled = self.norm(x[:, 0])
        return self.classifier(pooled)


class CNNBiLSTMFusion(nn.Module):
    """
    Baseline 2：CNN-BiLSTM。

    说明：当前工程保存的是窗口级特征向量，不是原始时间序列。
    因此这里将 EEG、EMG、Pinch 视作长度为 3 的模态序列，先用 1D-CNN 提取局部模态组合特征，
    再用 BiLSTM 建模模态序列依赖。
    """

    def __init__(
        self,
        eeg_dim: int,
        emg_dim: int,
        pinch_dim: int,
        d_model: int = 256,
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projector = ModalProjector(eeg_dim, emg_dim, pinch_dim, d_model, dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor) -> torch.Tensor:
        tokens = self.projector(eeg, emg, pinch)  # [B, 3, d]
        x = tokens.transpose(1, 2)                # [B, d, 3]
        x = self.conv(x).transpose(1, 2)          # [B, 3, d]
        out, _ = self.lstm(x)                     # [B, 3, 2h]
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNFusion(nn.Module):
    """
    Baseline 3：TCN-style Fusion。

    说明：这里是在三模态 token 序列上使用 TCN 残差卷积，作为时序/序列建模强基线。
    """

    def __init__(
        self,
        eeg_dim: int,
        emg_dim: int,
        pinch_dim: int,
        d_model: int = 256,
        num_classes: int = 2,
        levels: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projector = ModalProjector(eeg_dim, emg_dim, pinch_dim, d_model, dropout)
        blocks = []
        for i in range(levels):
            blocks.append(TemporalBlock(d_model, kernel_size=3, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor) -> torch.Tensor:
        tokens = self.projector(eeg, emg, pinch)  # [B, 3, d]
        x = tokens.transpose(1, 2)                # [B, d, 3]
        x = self.tcn(x).transpose(1, 2)           # [B, 3, d]
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class ConcatMLP(nn.Module):
    """补充基线：特征拼接 + MLP。可用于复现/核对静态拼接融合。"""

    def __init__(self, eeg_dim: int, emg_dim: int, pinch_dim: int, d_model: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        in_dim = eeg_dim + emg_dim + pinch_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([eeg, emg, pinch], dim=-1))


class LateFusionMLP(nn.Module):
    """补充基线：三模态分别分类，最后平均 logits，模拟晚期融合。"""

    def __init__(self, eeg_dim: int, emg_dim: int, pinch_dim: int, d_model: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        def branch(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        self.eeg_branch = branch(eeg_dim)
        self.emg_branch = branch(emg_dim)
        self.pinch_branch = branch(pinch_dim)

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor) -> torch.Tensor:
        logits = self.eeg_branch(eeg) + self.emg_branch(emg) + self.pinch_branch(pinch)
        return logits / 3.0


def build_baseline_model(
    model_name: str,
    eeg_dim: int,
    emg_dim: int,
    pinch_dim: int,
    num_classes: int,
    d_model: int = 256,
    dropout: float = 0.3,
    nhead: int = 4,
) -> nn.Module:
    """统一构建函数，训练脚本中直接调用。"""
    name = model_name.lower()
    if name in {"transformer", "mm_transformer", "multimodal_transformer"}:
        return MultimodalTransformerEncoder(eeg_dim, emg_dim, pinch_dim, d_model=d_model, num_classes=num_classes, nhead=nhead, dropout=dropout)
    if name in {"cnn_bilstm", "bilstm", "cnn-lstm", "cnn_lstm"}:
        return CNNBiLSTMFusion(eeg_dim, emg_dim, pinch_dim, d_model=d_model, num_classes=num_classes, dropout=dropout)
    if name in {"tcn", "tcn_fusion"}:
        return TCNFusion(eeg_dim, emg_dim, pinch_dim, d_model=d_model, num_classes=num_classes, dropout=dropout)
    if name in {"concat", "concat_mlp", "mlp"}:
        return ConcatMLP(eeg_dim, emg_dim, pinch_dim, d_model=d_model, num_classes=num_classes, dropout=dropout)
    if name in {"late", "late_fusion"}:
        return LateFusionMLP(eeg_dim, emg_dim, pinch_dim, d_model=d_model, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
