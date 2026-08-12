import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from generate_sample_data import generate_unsw_nb15_sample
from src.preprocessing import prepare_train_test_split, prepare_official_split
from src.ctgan_balancer import CTGANBalancer
from src.model import get_model
from src.utils import compute_metrics, save_experiment_results, print_paper_comparison_table

def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    balance_suffix = "balanced" if args.balance else "raw"
    exp_name = f"{args.model}_{balance_suffix}"
    print(f"\n==================================================")
    print(f"    RUNNING EXPERIMENT: {exp_name.upper()}")
    print(f"==================================================")

    # Step 1: Load Datasets (Check if official test set is provided)
    data_path = args.dataset
    if not os.path.exists(data_path):
        print(f"Dataset path '{data_path}' not found. Generating sample benchmark dataset...")
        df = generate_unsw_nb15_sample(output_path=data_path, num_samples=3000)
    else:
        print(f"Loading training dataset from: {data_path}")
        df = pd.read_csv(data_path)

    has_official_test = args.test_dataset and os.path.exists(args.test_dataset)

    # Step 2: Data Balancing using CTGAN (Applied to Training Set ONLY)
    if args.balance:
        print("\n--- STAGE: CTGAN Data Balancing (Applied to Training Set Only) ---")
        balancer = CTGANBalancer(target_col=args.target_col, epochs=args.ctgan_epochs)
        df = balancer.balance_dataset(df)

    # Step 3: Preprocessing & Min-Max 0-255 Normalization
    print("\n--- STAGE: Preprocessing & 0-255 Min-Max Normalization ---")
    if has_official_test:
        print(f"Loading official testing set from: {args.test_dataset}")
        test_df = pd.read_csv(args.test_dataset)
        X_train, X_test, y_train, y_test, preprocessor, zero_day_df = prepare_official_split(
            df, test_df, target_col=args.target_col, hide_class=args.hide_class
        )
    else:
        print("Using 80/20 Stratified Train/Test split on training file...")
        X_train, X_test, y_train, y_test, preprocessor, zero_day_df = prepare_train_test_split(
            df, target_col=args.target_col, test_size=args.test_size, hide_class=args.hide_class
        )

    num_features = X_train.shape[1]
    num_classes = len(preprocessor.target_encoder.classes_)
    class_names = preprocessor.target_encoder.classes_

    print(f"Features: {num_features} | Classes ({num_classes}): {list(class_names)}")
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Step 4: Prepare Data Loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Step 5: Instantiate Model
    model = get_model(args.model, input_features=num_features, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Step 6: Training Loop
    print(f"\n--- STAGE: Training Model for {args.epochs} Epochs ---")
    train_accs, train_losses = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                outputs, _, _ = model(X_batch)
            else:
                outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch}/{args.epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%")

    # Step 7: Evaluation on Test Set
    print("\n--- STAGE: Evaluation on Real Test Set ---")
    model.eval()
    y_preds, y_trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                outputs, _, _ = model(X_batch)
            else:
                outputs = model(X_batch)

            _, predicted = torch.max(outputs, 1)
            y_preds.extend(predicted.cpu().numpy())
            y_trues.extend(y_batch.numpy())

    metrics = compute_metrics(y_trues, y_preds)
    metrics['train_accuracy'] = train_accs[-1] if train_accs else 0.0

    print(f"\n✅ FINAL TRAIN ACCURACY: {metrics['train_accuracy'] * 100:.2f}%")
    print(f"✅ FINAL TEST ACCURACY : {metrics['accuracy'] * 100:.2f}%")

    # Step 8: LOCO Zero-Day Evaluation (If hide_class specified)
    if zero_day_df is not None and len(zero_day_df) > 0:
        print("\n======================================================================")
        print(f"         LOCO ZERO-DAY THREAT TEST: Hidden Class '{args.hide_class}'")
        print("======================================================================")
        X_zd, _ = preprocessor.transform(zero_day_df)
        X_zd_tensor = torch.tensor(X_zd, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                logits, is_zero_day, recon_loss = model(X_zd_tensor)
                detected = is_zero_day.sum().item()
                total_zd = len(X_zd_tensor)
                print(f"🛡️ DUAL-ENGINE ZERO-DAY GUARD RESULTS:")
                print(f"   Hidden Attack Tested        : '{args.hide_class}' ({total_zd} samples)")
                print(f"   Reconstruction Threshold (tau): {model.tau_threshold}")
                print(f"   Avg Reconstruction Loss     : {recon_loss.mean().item():.4f}")
                print(f"   Zero-Day Threats Detected   : {detected} / {total_zd} ({detected/total_zd*100:.2f}%)")
            else:
                outputs = model(X_zd_tensor)
                _, preds = torch.max(outputs, 1)
                normal_idx = 0
                if 'Normal' in class_names:
                    normal_idx = list(class_names).index('Normal')
                misclassified_as_normal = (preds == normal_idx).sum().item()
                total_zd = len(X_zd_tensor)
                print(f"❌ SUPERVISED MODEL ZERO-DAY RESULTS (No Engine 1):")
                print(f"   Hidden Attack Tested        : '{args.hide_class}' ({total_zd} samples)")
                print(f"   Misclassified as Normal     : {misclassified_as_normal} / {total_zd} ({misclassified_as_normal/total_zd*100:.2f}%)")
        print("======================================================================\n")

    # Step 9: Save & Organize Results
    save_experiment_results(exp_name, metrics, y_trues, y_preds, train_accs, train_losses, class_names)

    # Step 10: Print Side-by-Side Paper Comparison
    epoch_key = f"{args.epochs}epoch"
    print_paper_comparison_table(args.dataset_name, args.model.capitalize(), epoch_key, metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IoT Network Intrusion Detection System Training")
    parser.add_argument('--dataset', type=str, default='data/unsw_nb15_sample.csv', help='Path to training dataset CSV')
    parser.add_argument('--test_dataset', type=str, default=None, help='Path to official testing dataset CSV (optional)')
    parser.add_argument('--dataset_name', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018', 'CIC-IOT2023'], help='Dataset benchmark name')
    parser.add_argument('--model', type=str, default='proposed', choices=['proposed', 'dual-engine', 'cnn', 'resnet', 'resnet-gru'], help='Model architecture to evaluate')
    parser.add_argument('--target_col', type=str, default='attack_cat', help='Target label column name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (50, 100, 200)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set split ratio')
    parser.add_argument('--balance', action='store_true', help='Enable CTGAN data balancing')
    parser.add_argument('--ctgan_epochs', type=int, default=10, help='CTGAN training epochs')
    parser.add_argument('--hide_class', type=str, default=None, help='Class name to hide for LOCO Zero-Day simulation (e.g. Worms)')

    args = parser.parse_args()
    train_and_evaluate(args)

