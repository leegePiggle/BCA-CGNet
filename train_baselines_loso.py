"""
train_baselines_loso.py

用于在 FinalFusion 工程中跑主流多模态 baseline：
    - Multimodal Transformer Encoder
    - CNN-BiLSTM
    - TCN
    - Concat-MLP
    - Late Fusion MLP

特点：
1. 支持 PyTorch 2.6 的 weights_only=False 加载；
2. 支持 EEG 文件是纯 Tensor，也支持 dict['features']；
3. 默认使用 LOSO（Leave-One-Subject-Out）按受试者划分；
4. 输出时间片段级指标与受试者级多数投票指标；
5. 保存每个样本预测结果和 summary.csv。

注意：如果 EEG/EMG/Pinch 三模态样本数不一致，说明需要先做三模态对齐。
脚本会直接报错，避免错误地把未对齐样本用于论文实验。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from baseline_models import build_baseline_model


class FusionDatasetWithSubject(Dataset):
    def __init__(self, eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor, labels: torch.Tensor, subjects: Iterable[str]):
        self.eeg = eeg.float()
        self.emg = emg.float()
        self.pinch = pinch.float()
        self.labels = labels.long()
        self.subjects = np.asarray(list(subjects))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.eeg[idx], self.emg[idx], self.pinch[idx], self.labels[idx], self.subjects[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def torch_load_trusted(path: str) -> Any:
    """兼容 PyTorch 2.6：本地可信文件可使用 weights_only=False。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_features(obj: Any, name: str) -> torch.Tensor:
    if torch.is_tensor(obj):
        feat = obj
    elif isinstance(obj, dict):
        for key in ["features", "feat", "x", "data"]:
            if key in obj:
                feat = obj[key]
                break
        else:
            raise KeyError(f"{name} 文件是 dict，但没有找到 features/feat/x/data 键。现有 keys={list(obj.keys())}")
    else:
        raise TypeError(f"{name} 文件类型不支持: {type(obj)}")
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    if feat.dim() > 2:
        feat = feat.reshape(feat.shape[0], -1)
    return feat.float()


def find_first_key(obj: Any, keys: List[str]):
    if not isinstance(obj, dict):
        return None
    for k in keys:
        if k in obj:
            return obj[k]
    return None


def get_labels(task: str, eeg_obj: Any, emg_obj: Any, pinch_obj: Any) -> torch.Tensor:
    if task == "binary":
        candidate_keys = ["labels_binary", "binary_labels", "label_binary", "labels", "y"]
    elif task == "stage":
        candidate_keys = ["labels_stage", "stage_labels", "label_stage", "brunnstrom_labels", "labels_4class", "labels4", "labels", "y"]
    else:
        raise ValueError("task must be binary or stage")

    # 标签优先从 pinch 文件取，其次 EMG，最后 EEG
    for obj_name, obj in [("pinch", pinch_obj), ("emg", emg_obj), ("eeg", eeg_obj)]:
        labels = find_first_key(obj, candidate_keys)
        if labels is not None:
            labels = torch.as_tensor(labels).long().view(-1)
            unique = sorted(labels.unique().tolist())
            mapping = {old: new for new, old in enumerate(unique)}
            # 如果标签不是从 0 开始，自动重映射到 0..C-1
            remapped = torch.tensor([mapping[int(v)] for v in labels.tolist()], dtype=torch.long)
            print(f"Labels loaded from {obj_name}; original unique={unique}; mapping={mapping}")
            return remapped
    raise KeyError(f"没有找到 {task} 标签。请检查 pinch/emg/eeg 的 keys。")


def get_subject_ids(eeg_obj: Any, emg_obj: Any, pinch_obj: Any, n: int) -> np.ndarray:
    candidate_keys = ["subject_ids", "subject_id", "subjects", "sub_ids", "participant_ids"]
    for obj_name, obj in [("pinch", pinch_obj), ("emg", emg_obj), ("eeg", eeg_obj)]:
        subs = find_first_key(obj, candidate_keys)
        if subs is not None:
            subs = np.asarray([str(s) for s in list(subs)])
            if len(subs) == n:
                print(f"Subject IDs loaded from {obj_name}; subjects={len(np.unique(subs))}")
                return subs
            print(f"Warning: {obj_name} subject_ids length={len(subs)} != N={n}; skip")
    raise KeyError("没有找到长度匹配的 subject_ids。LOSO 必须有受试者编号。")


def validate_same_length(eeg: torch.Tensor, emg: torch.Tensor, pinch: torch.Tensor, labels: torch.Tensor, debug_truncate: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [eeg.size(0), emg.size(0), pinch.size(0), len(labels)]
    if len(set(lengths)) == 1:
        return eeg, emg, pinch, labels

    msg = (
        f"三模态样本数不一致：EEG={lengths[0]}, EMG={lengths[1]}, Pinch={lengths[2]}, Labels={lengths[3]}。\n"
        "这说明当前文件还没有完成最终对齐，不能直接用于正式论文实验。\n"
        "请先根据 file_paths/window_starts/original_indices 或已有 check_alignment.py 生成对齐后的三模态特征。"
    )
    if not debug_truncate:
        raise ValueError(msg)

    min_n = min(lengths)
    print("\n[DEBUG ONLY] " + msg)
    print(f"[DEBUG ONLY] 临时截断到前 {min_n} 个样本，仅用于调试代码能否运行，不能写入论文。\n")
    return eeg[:min_n], emg[:min_n], pinch[:min_n], labels[:min_n]


def compute_class_weights(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * torch.clamp(counts, min=1.0))
    return weights.to(device)


def unpack_logits(output: Any) -> torch.Tensor:
    # 兼容 proposed model 返回 (logits, fused, gate_weights)
    if isinstance(output, tuple):
        return output[0]
    return output


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int], List[str]]:
    model.eval()
    all_preds, all_labels, all_subjects = [], [], []
    with torch.no_grad():
        for eeg, emg, pinch, labels, subjects in loader:
            eeg = eeg.to(device)
            emg = emg.to(device)
            pinch = pinch.to(device)
            logits = unpack_logits(model(eeg, emg, pinch))
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_subjects.extend([str(s) for s in subjects])
    return all_labels, all_preds, all_subjects


def metric_dict(y_true: List[int], y_pred: List[int], num_classes: int, prefix: str) -> Dict[str, float]:
    labels = list(range(num_classes))
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        f"{prefix}_acc": float(acc),
        f"{prefix}_precision_macro": float(p_macro),
        f"{prefix}_recall_macro": float(r_macro),
        f"{prefix}_f1_macro": float(f_macro),
        f"{prefix}_precision_weighted": float(p_weight),
        f"{prefix}_recall_weighted": float(r_weight),
        f"{prefix}_f1_weighted": float(f_weight),
    }


def majority_vote_by_subject(y_true: List[int], y_pred: List[int], subjects: List[str]) -> Tuple[List[int], List[int], List[str]]:
    true_subj, pred_subj, subj_list = [], [], []
    for s in sorted(set(subjects)):
        idx = [i for i, x in enumerate(subjects) if x == s]
        true_label = Counter([y_true[i] for i in idx]).most_common(1)[0][0]
        pred_label = Counter([y_pred[i] for i in idx]).most_common(1)[0][0]
        subj_list.append(s)
        true_subj.append(true_label)
        pred_subj.append(pred_label)
    return true_subj, pred_subj, subj_list


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    train_labels: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_accum_steps: int,
) -> nn.Module:
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels, num_classes, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, (eeg, emg, pinch, labels, _) in enumerate(train_loader):
            eeg = eeg.to(device)
            emg = emg.to(device)
            pinch = pinch.to(device)
            labels = labels.to(device)
            logits = unpack_logits(model(eeg, emg, pinch))
            loss = criterion(logits, labels) / max(1, grad_accum_steps)
            loss.backward()
            running_loss += float(loss.item()) * max(1, grad_accum_steps)

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:03d}/{epochs} | train_loss={running_loss / max(1, len(train_loader)):.4f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./SingleModelFeaturePreprocess")
    parser.add_argument("--eeg_path", type=str, default=None)
    parser.add_argument("--emg_path", type=str, default=None)
    parser.add_argument("--pinch_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "cnn_bilstm", "tcn", "concat_mlp", "late_fusion"])
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "stage"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./baseline_results")
    parser.add_argument("--debug_truncate", action="store_true", help="仅调试使用：样本数不一致时截断到最小长度。不能用于论文。")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    eeg_path = args.eeg_path or os.path.join(args.base_dir, "eeg_features_merged", "eeg_features_all.pt")
    emg_path = args.emg_path or os.path.join(args.base_dir, "emg_features_merged", "emg_features_all.pt")
    pinch_path = args.pinch_path or os.path.join(args.base_dir, "pinch_features_merged", "pinch_features_all.pt")

    print("Loading features...")
    eeg_obj = torch_load_trusted(eeg_path)
    emg_obj = torch_load_trusted(emg_path)
    pinch_obj = torch_load_trusted(pinch_path)

    eeg = get_features(eeg_obj, "eeg")
    emg = get_features(emg_obj, "emg")
    pinch = get_features(pinch_obj, "pinch")
    labels = get_labels(args.task, eeg_obj, emg_obj, pinch_obj)
    eeg, emg, pinch, labels = validate_same_length(eeg, emg, pinch, labels, args.debug_truncate)
    subjects = get_subject_ids(eeg_obj, emg_obj, pinch_obj, n=len(labels))
    if len(subjects) != len(labels):
        if args.debug_truncate:
            subjects = subjects[:len(labels)]
        else:
            raise ValueError(f"subject_ids length={len(subjects)} != labels length={len(labels)}")

    num_classes = int(labels.max().item() + 1)
    print(f"Data shape: EEG={tuple(eeg.shape)}, EMG={tuple(emg.shape)}, Pinch={tuple(pinch.shape)}")
    print(f"Task={args.task}; num_classes={num_classes}; N={len(labels)}; subjects={len(np.unique(subjects))}")
    print("Class counts:", dict(Counter(labels.tolist())))

    os.makedirs(args.save_dir, exist_ok=True)
    run_name = f"{args.model}_{args.task}_loso"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    all_true, all_pred, all_subj = [], [], []
    unique_subjects = sorted(np.unique(subjects).tolist())

    for fold_id, test_subject in enumerate(unique_subjects, start=1):
        print(f"\n========== LOSO Fold {fold_id}/{len(unique_subjects)} | test_subject={test_subject} ==========")
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        train_dataset = FusionDatasetWithSubject(eeg[train_idx], emg[train_idx], pinch[train_idx], labels[train_idx], subjects[train_idx])
        test_dataset = FusionDatasetWithSubject(eeg[test_idx], emg[test_idx], pinch[test_idx], labels[test_idx], subjects[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = build_baseline_model(
            args.model,
            eeg_dim=eeg.shape[1],
            emg_dim=emg.shape[1],
            pinch_dim=pinch.shape[1],
            num_classes=num_classes,
            d_model=args.d_model,
            dropout=args.dropout,
            nhead=args.nhead,
        )
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    train_windows={len(train_idx)}, test_windows={len(test_idx)}, params={param_count:,}")

        model = train_one_fold(
            model=model,
            train_loader=train_loader,
            device=device,
            train_labels=labels[train_idx],
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_accum_steps=args.grad_accum_steps,
        )

        y_true, y_pred, y_subj = evaluate(model, test_loader, device)
        fold_metrics = metric_dict(y_true, y_pred, num_classes, prefix="window")
        print(f"    Fold window_acc={fold_metrics['window_acc']:.4f}, window_f1_macro={fold_metrics['window_f1_macro']:.4f}")

        all_true.extend(y_true)
        all_pred.extend(y_pred)
        all_subj.extend(y_subj)

        torch.save(model.state_dict(), os.path.join(run_dir, f"fold_{fold_id:02d}_{test_subject}.pth"))

    # 总体时间片段级指标：将所有 LOSO 测试预测汇总后计算
    window_metrics = metric_dict(all_true, all_pred, num_classes, prefix="window")

    # 受试者级多数投票指标
    subj_true, subj_pred, subj_names = majority_vote_by_subject(all_true, all_pred, all_subj)
    subject_metrics = metric_dict(subj_true, subj_pred, num_classes, prefix="subject")

    summary = {
        "model": args.model,
        "task": args.task,
        "num_classes": num_classes,
        "n_windows": len(all_true),
        "n_subjects": len(subj_names),
        **window_metrics,
        **subject_metrics,
    }

    print("\n================ SUMMARY ================")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # 保存样本级预测
    pred_df = pd.DataFrame({"subject_id": all_subj, "y_true": all_true, "y_pred": all_pred})
    pred_df.to_csv(os.path.join(run_dir, "window_predictions.csv"), index=False, encoding="utf-8-sig")

    # 保存受试者级预测
    subj_df = pd.DataFrame({"subject_id": subj_names, "y_true": subj_true, "y_pred": subj_pred})
    subj_df.to_csv(os.path.join(run_dir, "subject_predictions.csv"), index=False, encoding="utf-8-sig")

    # 保存 summary
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {run_dir}")
    print("Window confusion matrix:\n", confusion_matrix(all_true, all_pred, labels=list(range(num_classes))))
    print("Subject confusion matrix:\n", confusion_matrix(subj_true, subj_pred, labels=list(range(num_classes))))


if __name__ == "__main__":
    main()
