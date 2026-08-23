import os
import numpy as np
import torch
from collections import Counter

# ============================================================
# 1. 路径设置
# ============================================================
# 这里必须是包含 X.npy、y_binary.npy、y_stage.npy、sub_ids.npy 的 output_data 文件夹
DATA_OUTPUT_DIR = r"F:\李师姐小论文\Pinch\output_data"

# 输出文件建议保存到 FinalFusion 的 pinch_features_merged 中
SAVE_PATH = r"E:\111研究生\论文\李师姐论文\FinalFusion\SingleModelFeaturePreprocess\pinch_features_merged\pinch_window_features.pt"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

X_PATH = os.path.join(DATA_OUTPUT_DIR, "X.npy")
Y_BIN_PATH = os.path.join(DATA_OUTPUT_DIR, "y_binary.npy")
Y_STAGE_PATH = os.path.join(DATA_OUTPUT_DIR, "y_stage.npy")
SUB_PATH = os.path.join(DATA_OUTPUT_DIR, "sub_ids.npy")


# ============================================================
# 2. 参数设置
# ============================================================
FS = 30                  # 捏力采样率 30 Hz
DROP_FIRST_SEC = 2       # 去掉前 2 s 发力爬升段
WINDOW_SEC = 2           # 2 s 窗口
STRIDE_SEC = 1           # 50% overlap，即 1 s 步长

DROP_FIRST_PTS = FS * DROP_FIRST_SEC    # 60
WINDOW_PTS = FS * WINDOW_SEC            # 60
STRIDE_PTS = FS * STRIDE_SEC            # 30


# ============================================================
# 3. 工具函数
# ============================================================
def ensure_channel_first(X):
    """
    将 X 统一为 [N, C, T]
    当前你的 X 是 [111, 4, 360]，会保持不变。
    """
    if X.ndim != 3:
        raise ValueError(f"X should be 3D, but got shape {X.shape}")

    if X.shape[1] <= 16:
        return X  # [N, C, T]
    elif X.shape[2] <= 16:
        return np.transpose(X, (0, 2, 1))  # [N, T, C] -> [N, C, T]
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {X.shape}")


def entropy_1d(x, bins=10, eps=1e-8):
    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(np.float32) / (np.sum(hist) + eps)
    return -np.sum(p * np.log(p + eps))


def corr_with_time(x, eps=1e-8):
    t = np.arange(len(x), dtype=np.float32)
    t = t - np.mean(t)
    y = x.astype(np.float32) - np.mean(x)
    denom = np.sqrt(np.sum(t ** 2) * np.sum(y ** 2)) + eps
    return np.sum(t * y) / denom


def channel_features(x):
    """
    对单个通道的 2s 捏力窗口提取 32 个统计/动态特征。
    4 个通道 × 32 = 128 维。
    """
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x)
    eps = 1e-8

    if len(dx) == 0:
        dx = np.array([0.0], dtype=np.float32)

    t = np.arange(len(x), dtype=np.float32)

    try:
        slope = np.polyfit(t, x, 1)[0]
    except Exception:
        slope = 0.0

    zero_cross_rate = np.mean((x[:-1] * x[1:]) < 0) if len(x) > 1 else 0.0
    diff_sign_change_rate = np.mean((dx[:-1] * dx[1:]) < 0) if len(dx) > 1 else 0.0

    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(x)
    else:
        auc = np.trapz(x)

    feats = [
        np.mean(x),                                  # 1 均值
        np.std(x),                                   # 2 标准差
        np.var(x),                                   # 3 方差
        np.min(x),                                   # 4 最小值
        np.max(x),                                   # 5 最大值
        np.max(x) - np.min(x),                       # 6 极差
        np.median(x),                                # 7 中位数
        np.percentile(x, 10),                        # 8
        np.percentile(x, 25),                        # 9
        np.percentile(x, 75),                        # 10
        np.percentile(x, 90),                        # 11
        np.percentile(x, 75) - np.percentile(x, 25), # 12 四分位距
        np.sqrt(np.mean(x ** 2)),                    # 13 RMS
        np.mean(x ** 2),                             # 14 平均能量
        np.sum(x ** 2),                              # 15 总能量
        np.mean(np.abs(x)),                          # 16 绝对均值
        np.max(np.abs(x)),                           # 17 最大绝对值
        np.std(x) / (np.abs(np.mean(x)) + eps),      # 18 变异系数
        slope,                                       # 19 趋势斜率
        corr_with_time(x),                           # 20 与时间相关性
        auc,                                         # 21 面积
        x[0],                                        # 22 起点
        x[-1],                                       # 23 终点
        x[-1] - x[0],                                # 24 起终差
        np.mean(dx),                                 # 25 一阶差分均值
        np.std(dx),                                  # 26 一阶差分标准差
        np.min(dx),                                  # 27
        np.max(dx),                                  # 28
        np.mean(np.abs(dx)),                         # 29 差分绝对均值
        np.sqrt(np.mean(dx ** 2)),                   # 30 差分 RMS
        zero_cross_rate,                             # 31 过零率
        diff_sign_change_rate,                       # 32 差分符号变化率
    ]

    return np.asarray(feats, dtype=np.float32)


def extract_window_feature(window):
    """
    window: [C, 60]
    return: [128]
    """
    feats = []
    for c in range(window.shape[0]):
        feats.append(channel_features(window[c]))

    feats = np.concatenate(feats, axis=0)

    # 正常情况下 4 通道 × 32 = 128。
    # 如果通道数不是 4，则做截断/补零，保证输出 128 维。
    if len(feats) > 128:
        feats = feats[:128]
    elif len(feats) < 128:
        feats = np.pad(feats, (0, 128 - len(feats)), mode="constant")

    return feats.astype(np.float32)


# ============================================================
# 4. 主函数：trial-level -> window-level
# ============================================================
def main():
    print("Loading pinch X.npy ...")

    X = np.load(X_PATH, allow_pickle=True)
    y_binary = np.load(Y_BIN_PATH, allow_pickle=True)
    y_stage = np.load(Y_STAGE_PATH, allow_pickle=True)
    sub_ids = np.load(SUB_PATH, allow_pickle=True)

    X = ensure_channel_first(X).astype(np.float32)

    N, C, T = X.shape

    print("=" * 80)
    print("Original X shape:", X.shape)
    print("Binary label distribution:", Counter(y_binary.tolist()))
    print("Stage label distribution:", Counter(y_stage.tolist()))
    print("Subject distribution:")
    for k, v in Counter([str(s) for s in sub_ids]).most_common():
        print(k, v)

    window_features = []
    raw_windows = []
    labels_binary = []
    labels_stage = []
    subject_ids = []
    trial_indices = []
    window_indices = []
    window_start_valid_sec = []
    window_start_original_sec = []

    for trial_idx in range(N):
        trial = X[trial_idx]  # [C, T]

        # 当前你的 T=360，对应 12s。
        # 按论文逻辑：去掉前2s，保留后10s。
        if T >= 360:
            valid_trial = trial[:, DROP_FIRST_PTS:]  # [C, 300]
            original_offset_sec = DROP_FIRST_SEC
        elif T == 300:
            valid_trial = trial
            original_offset_sec = 0
        else:
            valid_trial = trial
            original_offset_sec = 0
            print(f"Warning: trial {trial_idx} has unusual length T={T}")

        valid_T = valid_trial.shape[1]

        local_window_idx = 0
        start = 0

        while start + WINDOW_PTS <= valid_T:
            win = valid_trial[:, start:start + WINDOW_PTS]  # [C, 60]

            feat = extract_window_feature(win)

            window_features.append(feat)
            raw_windows.append(win)

            labels_binary.append(int(y_binary[trial_idx]))
            labels_stage.append(int(y_stage[trial_idx]))
            subject_ids.append(str(sub_ids[trial_idx]))

            trial_indices.append(int(trial_idx))
            window_indices.append(int(local_window_idx))

            start_valid_sec = start / FS
            start_original_sec = original_offset_sec + start_valid_sec

            window_start_valid_sec.append(float(start_valid_sec))
            window_start_original_sec.append(float(start_original_sec))

            start += STRIDE_PTS
            local_window_idx += 1

    window_features = torch.tensor(np.asarray(window_features), dtype=torch.float32)
    raw_windows = torch.tensor(np.asarray(raw_windows), dtype=torch.float32)
    labels_binary = torch.tensor(labels_binary, dtype=torch.long)
    labels_stage = torch.tensor(labels_stage, dtype=torch.long)

    save_dict = {
        "features": window_features,                        # [N_window, 128]
        "raw_windows": raw_windows,                         # [N_window, 4, 60]
        "labels_binary": labels_binary,                     # [N_window]
        "labels_stage": labels_stage,                       # [N_window]
        "subject_ids": subject_ids,                         # list[str]
        "trial_indices": trial_indices,                     # 每个窗口来自 X.npy 的第几个 trial
        "window_indices": window_indices,                   # 每个 trial 内第几个窗口，0~8
        "window_start_valid_sec": window_start_valid_sec,   # 在稳定10s内的起始时间：0,1,...,8
        "window_start_original_sec": window_start_original_sec, # 在原始12s内的起始时间：2,3,...,10
        "fs": FS,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "drop_first_sec": DROP_FIRST_SEC,
        "source": "X.npy",
        "note": "Pinch window-level features generated from X.npy. Each 12s trial drops first 2s and slices the remaining 10s into 2s windows with 1s stride."
    }

    torch.save(save_dict, SAVE_PATH)

    print("=" * 80)
    print("Saved to:", SAVE_PATH)
    print("Window-level pinch features:", window_features.shape)
    print("Raw pinch windows:", raw_windows.shape)
    print("Binary labels:", labels_binary.shape, Counter(labels_binary.numpy().tolist()))
    print("Stage labels:", labels_stage.shape, Counter(labels_stage.numpy().tolist()))
    print("Unique subjects:", len(set(subject_ids)))
    print("Expected: if all 111 trials have T=360, windows = 111 * 9 = 999")


if __name__ == "__main__":
    main()