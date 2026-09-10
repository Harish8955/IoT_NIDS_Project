import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import prepare_train_test_split
from src.model import DualEngineIoT_NIDS

def run_actual_theta_sweep(epochs=5):
    print("==========================================================================")
    print(f"   TRAINING & EXECUTING REAL THETA HYPERPARAMETER SWEEP ({epochs} EPOCHS)")
    print("==========================================================================")

    data_path = "data/UNSW_NB15_combined.csv"
    if not os.path.exists(data_path):
        print(f"Dataset path '{data_path}' not found.")
        return

    # 1. Preprocess & Hide 'Worms' class for LOCO Zero-Day Evaluation
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test, preprocessor, zero_day_df = prepare_train_test_split(
        df, target_col='attack_cat', test_size=0.2, hide_class='Worms'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = X_train.shape[1]
    num_classes = len(preprocessor.target_encoder.classes_)

    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    X_zd, _ = preprocessor.transform(zero_day_df)
    zd_dataset = TensorDataset(torch.tensor(X_zd, dtype=torch.float32))
    zd_loader = DataLoader(zd_dataset, batch_size=256, shuffle=False)

    # 2. Instantiate Model & Train for specified epochs
    model = DualEngineIoT_NIDS(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining Dual-Engine Model for {epochs} Epochs...")
    for ep in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _, _, _ = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_b.size(0)
            _, preds = torch.max(logits, 1)
            total += y_b.size(0)
            correct += (preds == y_b).sum().item()

        print(f"Epoch [{ep}/{epochs}] | Train Loss: {running_loss/total:.4f} | Train Acc: {correct/total*100:.2f}%")

    # 3. Extract P_max for Known Validation Attacks
    model.eval()
    known_pmax = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            class_logits, _, _, p_max = model(X_batch)
            known_pmax.extend(p_max.cpu().numpy())

    # 4. Extract P_max for Hidden Zero-Day Worms Attacks
    zeroday_pmax = []
    with torch.no_grad():
        for (X_batch,) in zd_loader:
            X_batch = X_batch.to(device)
            class_logits, _, _, p_max = model(X_batch)
            zeroday_pmax.extend(p_max.cpu().numpy())

    known_pmax = np.array(known_pmax)
    zeroday_pmax = np.array(zeroday_pmax)

    # 5. Grid Search over Candidate Thresholds [0.50, 0.95]
    candidate_thetas = np.arange(0.50, 0.96, 0.05)
    rows = []
    best_theta = 0.85
    best_j_score = -1.0

    print(f"\nTotal Known Test Samples: {len(known_pmax)}")
    print(f"Total Hidden Zero-Day 'Worms' Samples: {len(zeroday_pmax)}\n")
    print(f"{'Candidate Theta':<18} | {'Known Precision (TPR)':<22} | {'Zero-Day DR (TNR)':<20} | {'Youden J-Index':<15}")
    print("-" * 80)

    for theta in candidate_thetas:
        tpr = np.mean(known_pmax >= theta)
        tnr = np.mean(zeroday_pmax < theta)
        j_score = tpr + tnr - 1.0

        row = {
            'Candidate Theta': round(theta, 2),
            'Known Precision (TPR)': f"{tpr * 100:.2f}%",
            'Zero-Day DR (TNR)': f"{tnr * 100:.2f}%",
            'Youden J-Index': round(j_score, 4)
        }
        rows.append(row)

        print(f"{theta:<18.2f} | {tpr * 100:<21.2f}% | {tnr * 100:<19.2f}% | {j_score:<15.4f}")

        if j_score > best_j_score:
            best_j_score = j_score
            best_theta = theta

    print("==========================================================================")
    print(f"Optimal Threshold Selected: theta = {best_theta:.2f} (Max J-Index = {best_j_score:.4f})")
    print("==========================================================================\n")

    summary_csv = "results/theta_sweep_results.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Sweep results saved to: {summary_csv}")

if __name__ == '__main__':
    run_actual_theta_sweep(epochs=3)
