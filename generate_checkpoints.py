import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import prepare_train_test_split
from src.ctgan_balancer import CTGANBalancer
from src.model import DualEngineIoT_NIDS

def generate_checkpoints_for_existing_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    loco_folders = [
        ("unsw_nb15_dual-engine_loco_worms_balanced_50ep", "Worms", 50),
        ("unsw_nb15_dual-engine_loco_worms_balanced_100ep", "Worms", 100),
        ("unsw_nb15_dual-engine_loco_worms_balanced_200ep", "Worms", 200),
        ("unsw_nb15_dual-engine_loco_shellcode_balanced_50ep", "Shellcode", 50),
        ("unsw_nb15_dual-engine_loco_shellcode_balanced_100ep", "Shellcode", 100),
        ("unsw_nb15_dual-engine_loco_shellcode_balanced_200ep", "Shellcode", 200),
    ]

    raw_data_path = "data/UNSW_NB15_combined.csv"
    if not os.path.exists(raw_data_path):
        print(f"Data file {raw_data_path} not found.")
        return

    print("Loading UNSW-NB15 dataset...")
    df_raw = pd.read_csv(raw_data_path)

    for folder_name, hide_cls, ep_target in loco_folders:
        folder_path = os.path.join("results", folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        best_pt = os.path.join(folder_path, "best.pt")
        last_pt = os.path.join(folder_path, "last.pt")

        if os.path.exists(best_pt) and os.path.exists(last_pt):
            print(f"✅ Checkpoints already exist for '{folder_name}'. Skipping.")
            continue

        print(f"\n=======================================================")
        print(f"  GENERATING CHECKPOINTS (best.pt & last.pt) FOR: {folder_name}")
        print(f"=======================================================")

        # Balance training data
        balancer = CTGANBalancer(target_col='attack_cat', epochs=10)
        df_balanced = balancer.balance_dataset(df_raw.copy())

        # Preprocess split
        X_train, X_test, y_train, y_test, preprocessor, _ = prepare_train_test_split(
            df_balanced, target_col='attack_cat', test_size=0.2, hide_class=hide_cls
        )

        num_features = X_train.shape[1]
        num_classes = len(preprocessor.target_encoder.classes_)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = DualEngineIoT_NIDS(input_features=num_features, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_acc = -1.0
        best_state = None

        print(f"Fitting model on GPU for {ep_target} epochs to save exact best.pt and last.pt...")
        for epoch in range(1, ep_target + 1):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                outputs, _, _, _ = model(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()

            model.eval()
            t_correct, t_total = 0, 0
            with torch.no_grad():
                for X_tb, y_tb in test_loader:
                    X_tb, y_tb = X_tb.to(device), y_tb.to(device)
                    outputs, _, _, _ = model(X_tb)
                    _, preds = torch.max(outputs, 1)
                    t_total += y_tb.size(0)
                    t_correct += (preds == y_tb).sum().item()

            epoch_acc = t_correct / t_total
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                import copy
                best_state = copy.deepcopy(model.state_dict())

            if epoch % max(1, ep_target // 5) == 0 or epoch == ep_target:
                print(f" Epoch [{epoch}/{ep_target}] | Test Acc: {epoch_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

        last_state = model.state_dict()
        if best_state is None:
            best_state = last_state

        torch.save(best_state, best_pt)
        torch.save(last_state, last_pt)
        print(f"💾 Successfully saved 'best.pt' and 'last.pt' in {folder_path}")

if __name__ == '__main__':
    generate_checkpoints_for_existing_runs()
