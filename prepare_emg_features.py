import os
import torch
import numpy as np

# ==========================================
# 配置
# ==========================================
FEAT_DIR = "./SingleModelFeature/EMGFeature"  # 肌电特征文件所在目录
OUTPUT_DIR = "./SingleModelFeaturePreprocess/emg_features_merged"  # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. 加载并合并五折特征
# ==========================================
all_features = []
all_labels = []
all_file_paths = []
all_window_starts = []
all_subject_ids = []

for fold in range(1, 6):
    file_path = os.path.join(FEAT_DIR, f"FusionFold{fold}_test_features.pt")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping fold {fold}")
        continue

    data = torch.load(file_path, map_location='cpu')
    all_features.append(data['features'])
    all_labels.append(data['labels'])
    all_file_paths.extend(data['file_paths'])
    all_window_starts.extend(data['window_starts'])
    all_subject_ids.extend(data['subject_ids'])

    print(f"Loaded Fold {fold}: {data['features'].shape[0]} samples")

# 拼接特征和标签
features = torch.cat(all_features, dim=0)  # [N_total, 64]
labels = torch.cat([lb.view(-1) for lb in all_labels], dim=0)  # [N_total]

print(f"\nTotal samples after merge: {features.shape[0]}")
print(f"Feature dimension: {features.shape[1]}")

# ==========================================
# 2. 构建按 (文件路径, 窗口起始时间) 排序的索引
#    便于与脑电特征对齐
# ==========================================
# 创建一个排序键列表
sort_keys = [(fp, ws) for fp, ws in zip(all_file_paths, all_window_starts)]
# 获取排序后的索引
sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

# 按排序索引重排所有数据
features_sorted = features[sorted_indices]
labels_sorted = labels[sorted_indices]
file_paths_sorted = [all_file_paths[i] for i in sorted_indices]
window_starts_sorted = [all_window_starts[i] for i in sorted_indices]
subject_ids_sorted = [all_subject_ids[i] for i in sorted_indices]

# ==========================================
# 3. 保存合并后的特征和元数据
# ==========================================
save_dict = {
    'features': features_sorted,  # 排序后的特征张量 [N, 64]
    'labels': labels_sorted,  # 标签 [N]
    'file_paths': file_paths_sorted,  # 每个窗口对应的原始文件路径
    'window_starts': window_starts_sorted,  # 窗口起始时间（秒）
    'subject_ids': subject_ids_sorted,  # 受试者 ID
    'original_indices': sorted_indices  # 原始加载顺序的索引（可选）
}

torch.save(save_dict, os.path.join(OUTPUT_DIR, "emg_features_all.pt"))
print(f"\nMerged features saved to {os.path.join(OUTPUT_DIR, 'emg_features_all.pt')}")

# 同时保存一份易读的 CSV 元数据（方便查看）
import csv

csv_path = os.path.join(OUTPUT_DIR, "emg_metadata.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_path', 'window_start_sec', 'subject_id', 'label'])
    for fp, ws, sid, lb in zip(file_paths_sorted, window_starts_sorted, subject_ids_sorted, labels_sorted.tolist()):
        writer.writerow([fp, ws, sid, lb])
print(f"Metadata CSV saved to {csv_path}")