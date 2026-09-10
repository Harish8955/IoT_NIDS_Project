import os
import json
import pandas as pd

harish_runs = [
    ('UNSW-NB15 Proposed Model (50 Ep)', 'unsw_nb15_proposed_balanced_50ep'),
    ('UNSW-NB15 Proposed Model (100 Ep)', 'unsw_nb15_proposed_balanced_100ep'),
    ('UNSW-NB15 Proposed Model (200 Ep)', 'unsw_nb15_proposed_balanced_200ep'),
    ('UNSW-NB15 1D-CNN Baseline (50 Ep)', 'unsw_nb15_cnn_balanced_50ep'),
    ('UNSW-NB15 1D-CNN Baseline (100 Ep)', 'unsw_nb15_cnn_balanced_100ep'),
    ('UNSW-NB15 1D-CNN Baseline (200 Ep)', 'unsw_nb15_cnn_balanced_200ep'),
    ('UNSW-NB15 ResNet Baseline (50 Ep)', 'unsw_nb15_resnet_balanced_50ep'),
    ('UNSW-NB15 ResNet Baseline (100 Ep)', 'unsw_nb15_resnet_balanced_100ep'),
    ('UNSW-NB15 ResNet Baseline (200 Ep)', 'unsw_nb15_resnet_balanced_200ep'),
    ('UNSW-NB15 ResNeSt Baseline (200 Ep)', 'unsw_nb15_resnest_balanced_200ep'),
    ('UNSW-NB15 ResNet-GRU Baseline (100 Ep)', 'unsw_nb15_resnet-gru_balanced_100ep'),
    ('UNSW-NB15 ResNet-GRU Baseline (200 Ep)', 'unsw_nb15_resnet-gru_balanced_200ep'),
]

rows = []
for label, folder in harish_runs:
    m_path = os.path.join('results', folder, 'metrics.json')
    if os.path.exists(m_path):
        with open(m_path, 'r') as f:
            data = json.load(f)
        train_acc = round(data.get('train_accuracy', 0.0) * 100, 2)
        test_acc = round(data.get('test_accuracy', data.get('accuracy', 0.0)) * 100, 2)
        f1_macro = round(data.get('f1_macro', 0.0) * 100, 2)
        far = round(data.get('false_alarm_rate', 0.0) * 100, 2)
        dr = round(data.get('detection_rate', 0.0) * 100, 2)
        rows.append({
            'Experiment Run': label,
            'Folder Name': folder,
            'Train Accuracy': f"{train_acc:.2f}%",
            'Test Accuracy': f"{test_acc:.2f}%",
            'Macro F1-Score': f"{f1_macro:.2f}%",
            'False Alarm Rate (FAR)': f"{far:.2f}%",
            'Detection Rate (DR)': f"{dr:.2f}%"
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
