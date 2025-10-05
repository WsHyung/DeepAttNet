import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from Deep4Net import Deep4AttNet, Deep4SelfAttNet, Deep4LiteTransNet
import copy
import random
from sklearn.metrics import f1_score
import json, itertools
from tqdm import tqdm
import hashlib, json

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def make_cfg_tag(cfg: dict) -> str:
    """Create tag for ckpt naming"""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}".replace('.', 'p')  # 0.08 -> 0p08
        return str(v)
    key_order = [
        "n_filters_time", "n_filters2", "n_filters3", "n_filters4",
        "hidden", "drop_prob", "filter_time_length",
        "pool_time_length", "pool_time_stride"
    ]
    parts = []
    for k in key_order:
        if k in cfg:
            parts.append(f"{k[:3]}{fmt(cfg[k])}")    # e.g.: n_f32, dro0p3
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    return "_".join(parts) + f"__{h}"

# Dataset Class
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Data load function
def load_data(folder_path, channels):
    data = []
    labels = []
    for label, subfolder in enumerate(["Relax", "Stress"]):  # 0: Relax, 1: Stress
        subfolder_path = os.path.join(folder_path, subfolder)
        for subject in sorted(os.listdir(subfolder_path)):
            subject_path = os.path.join(subfolder_path, subject)
            subject_data = np.expand_dims(np.load(subject_path), 0)
            data.append(subject_data[:, channels, :])
            labels.append(np.full(len(subject_data), label))
    return np.array(data), np.array(labels)


seed_everything(42)

# Load data & Ear-EEG channel selection
channels = [6, 7]
data_folder = "Preprocessed"
data, labels = load_data(data_folder, channels)

subjects = [f.split('/')[-1].split('.')[0] for f in os.listdir(os.path.join(data_folder, "Relax"))]

# 8-foldCV setting
kf = KFold(n_splits=8, shuffle=True, random_state=42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1) Grid search setting
# -----------------------------
def build_grid():
    return {
        "n_filters_time": [16, 32, 64],
        "filter_time_length": [0.5, 1.0],     # (= 0.5s, 1.0s)
        "pool_time_length": [5/125, 10/125],
        "pool_time_stride": [2/125, 5/125],

        "n_filters2":   [16, 32],
        "filter_length2":[20],
        "n_filters3":   [32, 64],
        "filter_length3":[20],
        "n_filters4":   [64,128],
        "filter_length4":[20],

        "hidden": [64, 128],
        "drop_prob": [0.3],
        "init_xavier": [0],
    }

def hp_from_HP_DEFAULT2(overrides: dict):
    from Deep4Net import HP_DEFAULT2 as BASE
    hp = copy.deepcopy(BASE)
    for k, v in overrides.items():
        if k in ["in_chans","epoch_time","sampling_rate","n_signal","n_classes"]:
            continue
        setattr(hp, k, v)
    return hp

# -----------------------------
# 8-Fold training & Validation for single sweep
# -----------------------------
def run_cv_for_hp(hp_ns, device, cfg_tag, train_cfg=None):
    # train_cfg: Training related settings
    if train_cfg is None:
        train_cfg = dict(batch_train=16, batch_val=8, max_epoch=200,
                         lr=1e-3, patience=30)

    fold_val_f1 = []
    fold_test_f1 = []
    fold_test_acc = []

    kf = KFold(n_splits=8, shuffle=True, random_state=42)
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(subjects)):
        # Train:Validation split (6:1)
        print(test_idx)
        train_idx = train_val_idx[:int(len(train_val_idx) * 6 / 7)]
        val_idx = train_val_idx[int(len(train_val_idx) * 6 / 7):]

        train_subjects = [subjects[i] for i in train_idx]
        val_subjects = [subjects[i] for i in val_idx]
        test_subjects = [subjects[i] for i in test_idx]

        # Train, Validation, Test dataset
        train_data, train_labels = [], []
        val_data, val_labels = [], []
        test_data, test_labels = [], []

        for i, subject in enumerate(subjects):
            if subject in train_subjects:
                train_data.append(data[i])
                train_labels.append(labels[i])
                train_data.append(data[i+32])
                train_labels.append(labels[i+32])
            elif subject in val_subjects:
                val_data.append(data[i])
                val_labels.append(labels[i])
                val_data.append(data[i+32])
                val_labels.append(labels[i+32])
            elif subject in test_subjects:
                test_data.append(data[i])
                test_labels.append(labels[i])
                test_data.append(data[i+32])
                test_labels.append(labels[i+32])

        # 배열로 변환
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        val_data = np.concatenate(val_data, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        train_data = np.expand_dims(train_data, 1)
        val_data = np.expand_dims(val_data, 1)
        test_data = np.expand_dims(test_data, 1)


        train_dataset = EEGDataset(train_data, train_labels)
        val_dataset = EEGDataset(val_data, val_labels)
        test_dataset = EEGDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


        # Model created with hp
        model = Deep4AttNet(hp=hp_ns).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=train_cfg["lr"],
                                eps=1e-8, amsgrad=False)

        # --- Model Training ---
        best_val_loss = np.inf; epochs_no_improve = 0
        for epoch in tqdm(range(train_cfg["max_epoch"])):
            model.train(); train_loss = 0.0
            for inputs, lab in train_loader:
                inputs, lab = inputs.to(device), lab.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, lab)
                loss.backward(); optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval(); val_loss = 0.0; v_pred=[]; v_true=[]
            with torch.no_grad():
                for inputs, lab in val_loader:
                    inputs, lab = inputs.to(device), lab.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, lab)
                    val_loss += loss.item()
                    v_pred.append(torch.argmax(outputs, dim=1).cpu())
                    v_true.append(lab.cpu())
            v_pred = torch.cat(v_pred); v_true = torch.cat(v_true)
            val_f1 = f1_score(v_true.numpy(), v_pred.numpy(), average='macro')

            if val_loss < best_val_loss:
                best_val_loss = val_loss; epochs_no_improve = 0
                ckpt_path = f"temp_fold{fold}__{cfg_tag}.pth"
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == train_cfg["patience"]:
                    break

        # --- Test (with best ckpt) ---
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval(); t_pred=[]; t_true=[]; correct=0; total=0
        with torch.no_grad():
            for inputs, lab in test_loader:
                inputs, lab = inputs.to(device), lab.to(device)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == lab).sum().item(); total += lab.size(0)
                t_pred.append(pred.cpu()); t_true.append(lab.cpu())
        t_pred = torch.cat(t_pred); t_true = torch.cat(t_true)
        test_f1 = f1_score(t_true.numpy(), t_pred.numpy(), average='macro')
        test_acc = correct/total

        fold_val_f1.append(val_f1)
        fold_test_f1.append(test_f1)
        fold_test_acc.append(test_acc)

    return np.mean(fold_val_f1), np.mean(fold_test_f1), np.mean(fold_test_acc), fold_test_f1, fold_test_acc

# -----------------------------
# Grid Loop
# -----------------------------
def grid_search(device):
    grid = build_grid()
    keys = list(grid.keys())
    results = []
    best = {"val_f1": None, "hp": None, "test_f1": None, "test_acc": -1, "test_f1_std": None, "test_acc_std": None}

    for values in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, values))
        hp_ns = hp_from_HP_DEFAULT2(cfg)
        cfg_tag = make_cfg_tag(cfg)

        val_f1, test_f1, test_acc, fls, accs = run_cv_for_hp(hp_ns, device, cfg_tag)

        row = {"val_f1": val_f1, "test_f1": test_f1, "test_acc": test_acc, "test_f1_std" : np.std(fls), "test_acc_std" : np.std(accs)}
        row.update(cfg)
        results.append(row)

        print(f"[GRID] {cfg} -> val_f1={val_f1:.4f}, test_f1={test_f1:.4f}, test_acc={test_acc:.4f}")

        if test_acc > best["test_acc"]:
            best = {"val_f1": val_f1, "hp": cfg, "test_f1": test_f1, "test_acc": test_acc}

        # Save current sweep result
        import pandas as pd
        pd.DataFrame(results).to_csv("grid_results.csv", index=False)
        with open("best_config.json","w") as f:
            json.dump(best, f, indent=2)

    print("\n=== BEST (by mean test_acc) ===")
    print(json.dumps(best, indent=2))
    return best

if __name__ == "__main__":
    best = grid_search(device)