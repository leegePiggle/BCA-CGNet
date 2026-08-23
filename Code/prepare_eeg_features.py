import os
import torch
import json
import numpy as np
from collections import defaultdict

EEG_EXP_ROOT = "./SingleModelFeature/EEGFeature"  # 替换为实际路径
OUTPUT_DIR = "./SingleModelFeaturePreprocess/eeg_features_merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_features = []
all_indices = []
all_metadata = []

# for fold in range(5):
#     fold_path = os.path.join(EEG_EXP_ROOT, f"fold{fold}")
#     feat = torch.load(os.path.join(fold_path, "features.pt"))
#     idx = torch.load(os.path.join(fold_path, "sample_indices.pt"))
#     with open(os.path.join(fold_path, "metadata_test.json"), "r") as f:
#         meta = json.load(f)
#
#     all_features.append(feat)
#     all_indices.append(idx)
#     all_metadata.append(meta)
for fold in range(5):
    fold_path = os.path.join(EEG_EXP_ROOT, f"fold{fold}")
    feat = torch.load(os.path.join(fold_path, "features.pt")).float()  # 强制转为 float32
    idx = torch.load(os.path.join(fold_path, "sample_indices.pt"))
    with open(os.path.join(fold_path, "metadata_test.json"), "r") as f:
        meta = json.load(f)

    all_features.append(feat)
    all_indices.append(idx)
    all_metadata.append(meta)

# 合并
features = torch.cat(all_features, dim=0)  # [N_total, D_eeg]
indices = torch.cat(all_indices, dim=0)  # [N_total]

# 按原始索引排序
sorted_order = torch.argsort(indices)
features_sorted = features[sorted_order]  # 按索引从小到大排列
indices_sorted = indices[sorted_order]

# 取第一个折的 metadata 作为完整列表（所有折的 metadata_test.json 内容相同）
full_metadata = all_metadata[0]

# 构建路径到特征的映射字典
# 注意：metadata 中的 "path" 字段是脑电文件的绝对路径，我们需要用它来匹配肌电文件的对应试验
path_to_feat = {}
for i, meta in enumerate(full_metadata):
    path = meta["path"]
    path_to_feat[path] = features_sorted[i].numpy()  # 转为 numpy 方便后续加载

# 保存映射字典
np.save(os.path.join(OUTPUT_DIR, "eeg_path_to_feat.npy"), path_to_feat)
torch.save(features_sorted, os.path.join(OUTPUT_DIR, "eeg_features_all.pt"))
with open(os.path.join(OUTPUT_DIR, "eeg_metadata.json"), "w") as f:
    json.dump(full_metadata, f)

print(f"EEG features prepared. Total samples: {len(full_metadata)}")