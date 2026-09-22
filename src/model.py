import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. WANG ET AL. (2024) PROPOSED MODEL COMPONENTS
# =====================================================================

class SplitAttention1D(nn.Module):
    """
    Split-Attention Block as specified in Fig. 4:
    Input -> GlobalAveragePooling -> Dense -> Dense -> r-Softmax -> r-Outputs
    """
    def __init__(self, channels, radix=2, reduction_factor=4):
        super(SplitAttention1D, self).__init__()
        self.channels = channels
        self.radix = radix
        inter_channels = max(8, channels // reduction_factor)
        
        self.fc1 = nn.Linear(channels, inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(inter_channels, channels * radix)
        self.rsoftmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        # Global Average Pooling (aggregating spatial information)
        gap = x.mean(dim=-1)
        
        att = self.fc1(gap)
        att = self.relu(att)
        att = self.fc2(att)
        
        att = att.view(batch_size, self.radix, self.channels)
        att = self.rsoftmax(att)
        
        # Cross-channel soft attention weighting across radix paths
        weight = att.sum(dim=1).unsqueeze(-1)
        return x * weight

class ResNeStBlock1D(nn.Module):
    """
    Grouped Convolutional Residual Unit as specified in Fig. 3:
    Branch: Conv 1x1 -> Conv 1x3 (Grouped) -> Split-Attention -> Conv 1x1 + Residual
    """
    def __init__(self, in_channels, out_channels, radix=2, groups=2):
        super(ResNeStBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.split_attention = SplitAttention1D(out_channels, radix=radix)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.split_attention(out)
        out = self.bn3(self.conv3(out))
        out += residual
        return self.relu(out)

class IoT_NIDS_Net(nn.Module):
    """
    Proposed Model in Wang et al. (2024) [Fig. 1 & Fig. 2]:
    1. Spatial Feature Extraction: 3 Conv1d (1x3) groups + BatchNorm + MaxPool
    2. ResBlock: 1D ResNeSt Grouped Residual Unit with Split-Attention
    3. Temporal Feature Extraction: BiGRU
    4. Classification: Dropout -> Fully Connected Softmax
    """
    def __init__(self, input_features, num_classes, hidden_gru=64, dropout_rate=0.5):
        super(IoT_NIDS_Net, self).__init__()
        
        # Preliminary 1D Spatial Feature Extraction (Fig. 2 in paper)
        self.spatial_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        )
        
        # 1D ResNeSt Block (Fig. 3 in paper)
        self.resblock = ResNeStBlock1D(in_channels=128, out_channels=128, radix=2, groups=2)
        
        # BiGRU Temporal Feature Extractor (Section 3.2.2 in paper)
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=hidden_gru,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_gru * 2, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (Batch, 1, Features)
            
        x = self.spatial_extractor(x)
        x = self.resblock(x)
        
        # Transpose for BiGRU: (Batch, Channels, Time) -> (Batch, Time, Channels)
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        
        # Global temporal pooling across time dimension
        context_vector = torch.mean(gru_out, dim=1)
        out = self.dropout(context_vector)
        return self.fc(out)


# =====================================================================
# 2. BASELINE MODELS (TABLES 12, 13, 14 IN WANG ET AL.)
# =====================================================================

class CNN1D(nn.Module):
    def __init__(self, input_features, num_classes):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.mean(x, dim=-1)
        return self.fc(x)

class ResNet1D(nn.Module):
    def __init__(self, input_features, num_classes):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.res_block = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        residual = x
        out = self.res_block(x) + residual
        out = F.relu(out)
        out = torch.mean(out, dim=-1)
        return self.fc(out)

class ResNetGRU1D(nn.Module):
    def __init__(self, input_features, num_classes):
        super(ResNetGRU1D, self).__init__()
        self.resnet = ResNet1D(input_features, num_classes=64)
        self.bigru = nn.GRU(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.resnet.conv1(x)
        residual = x
        x = F.relu(self.resnet.res_block(x) + residual)
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        out = torch.mean(gru_out, dim=1)
        return self.fc(out)


# =====================================================================
# 3. NOVEL EXTENSION: CONFIDENCE-AWARE DUAL-ENGINE ZERO-DAY GUARD
# =====================================================================

class Engine1_ZeroDayAutoencoder(nn.Module):
    """
    ENGINE 1: Deep Reconstruction Autoencoder (Zero-Day Guard).
    Trained ONLY on Benign Normal traffic.
    Architecture: N -> 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64 -> N
    Reconstructs 0-255 scaled features linearly without Sigmoid clamp.
    """
    def __init__(self, input_features, latent_dim=8):
        super(Engine1_ZeroDayAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_features)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def compute_anomaly_score(self, x):
        """Calculates MSE Reconstruction Loss per sample."""
        if x.dim() == 3:
            x = x.squeeze(1)
        reconstructed, latent = self.forward(x)
        loss = torch.mean((x - reconstructed) ** 2, dim=1)
        return loss, latent

class DualEngineIoT_NIDS(nn.Module):
    """
    CONFIDENCE-AWARE DUAL-ENGINE ARCHITECTURE:
    - Engine 1: Unsupervised Deep Autoencoder Guard (tau = 0.0191)
    - Engine 2: Supervised 1D ResNeSt + BiGRU Classifier (theta = 0.85)
    """
    def __init__(self, input_features, num_classes, tau_threshold=0.0191, theta_threshold=0.85):
        super(DualEngineIoT_NIDS, self).__init__()
        self.engine1_zero_day_guard = Engine1_ZeroDayAutoencoder(input_features)
        self.engine2_classifier = IoT_NIDS_Net(input_features, num_classes)
        self.tau_threshold = tau_threshold
        self.theta_threshold = theta_threshold

    def forward(self, x):
        recon_loss, latent = self.engine1_zero_day_guard.compute_anomaly_score(x)
        class_logits = self.engine2_classifier(x)
        probs = F.softmax(class_logits, dim=-1)
        p_max, _ = torch.max(probs, dim=-1)
        
        # Dual-Engine Gating Protocol (Section 1 Decision Matrix):
        # Triggered if Autoencoder flags anomaly OR Classifier is uncertain
        is_anomaly = recon_loss > self.tau_threshold
        is_low_confidence = p_max < self.theta_threshold
        is_zero_day = is_anomaly | is_low_confidence

        return class_logits, is_zero_day, recon_loss, p_max

    def predict_verdict(self, x, normal_class_idx=0):
        """
        Returns 3-way Security Verdict:
        - 0: BENIGN NORMAL TRAFFIC (L_recon <= tau AND P_max >= theta AND pred == normal)
        - 1: KNOWN ATTACK CLASS    (L_recon <= tau AND P_max >= theta AND pred != normal)
        - 2: ZERO-DAY THREAT ALERT (L_recon > tau OR P_max < theta)
        """
        class_logits, is_zero_day, recon_loss, p_max = self.forward(x)
        _, preds = torch.max(class_logits, dim=-1)
        
        verdicts = torch.full((x.size(0),), 2, dtype=torch.long, device=x.device)
        is_normal_recon = recon_loss <= self.tau_threshold
        is_high_conf = p_max >= self.theta_threshold
        
        # Benign Normal
        verdicts[is_normal_recon & is_high_conf & (preds == normal_class_idx)] = 0
        # Known Attack
        verdicts[is_normal_recon & is_high_conf & (preds != normal_class_idx)] = 1
        
        return verdicts, preds, p_max, recon_loss

def get_model(model_name, input_features, num_classes):
    m = model_name.lower()
    if m in ['proposed', '1d-resnest-bigru', 'resnest-bigru', 'resnest']:
        return IoT_NIDS_Net(input_features, num_classes)
    elif m in ['dual-engine', 'dual_engine', 'dual']:
        return DualEngineIoT_NIDS(input_features, num_classes)
    elif m == 'cnn':
        return CNN1D(input_features, num_classes)
    elif m == 'resnet':
        return ResNet1D(input_features, num_classes)
    elif m in ['resnet-gru', 'resnet-bigru']:
        return ResNetGRU1D(input_features, num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}'.")
```[cite: 3, 7, 8, 9]