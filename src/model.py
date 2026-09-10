import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. PROPOSED MODEL COMPONENTS (1D ResNeSt + BiGRU)
# =====================================================================

class SplitAttention1D(nn.Module):
    """1D Split-Attention Block (ResNeSt 1D extension)."""
    def __init__(self, channels, radis=2, reduction_factor=4):
        super(SplitAttention1D, self).__init__()
        self.channels = channels
        self.radis = radis
        inter_channels = max(8, channels // reduction_factor)
        
        self.fc1 = nn.Linear(channels, inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(inter_channels, channels * radis)
        self.rsoftmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        gap = x.mean(dim=-1)
        
        att = self.fc1(gap)
        att = self.relu(att)
        att = self.fc2(att)
        
        att = att.view(batch_size, self.radis, self.channels)
        att = self.rsoftmax(att)
        
        weight = att[:, 0, :].unsqueeze(-1)
        return x * weight

class ResNeStBlock1D(nn.Module):
    """1D Residual Grouped Conv Unit with Split-Attention."""
    def __init__(self, channels):
        super(ResNeStBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.split_attention = SplitAttention1D(channels)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.split_attention(out)
        out = self.bn3(self.conv3(out))
        out += residual
        return self.relu(out)

class IoT_NIDS_Net(nn.Module):
    """
    Proposed Model in Paper: 1D ResNeSt + BiGRU
    """
    def __init__(self, input_features, num_classes, hidden_gru=64, dropout_rate=0.5):
        super(IoT_NIDS_Net, self).__init__()
        self.conv_group1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True)
        )
        self.conv_group2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True)
        )
        self.conv_group3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.resnest_unit = ResNeStBlock1D(128)
        self.bigru = nn.GRU(
            input_size=128, hidden_size=hidden_gru, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_gru * 2, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_group1(x)
        x = self.conv_group2(x)
        x = self.conv_group3(x)
        x = self.resnest_unit(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        context_vector = torch.mean(gru_out, dim=1)
        out = self.dropout(context_vector)
        return self.fc(out)


# =====================================================================
# 2. BASELINE MODELS (Used in Paper Tables 12, 13, 14 for Comparison)
# =====================================================================

class CNN1D(nn.Module):
    """Standard 1D-CNN Baseline Model"""
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
    """Standard 1D-ResNet Baseline Model"""
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
    """ResNet-BiGRU Baseline Model"""
    def __init__(self, input_features, num_classes):
        super(ResNetGRU1D, self).__init__()
        self.resnet = ResNet1D(input_features, num_classes=64)
        self.bigru = nn.GRU(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Pass through Conv
        x = self.resnet.conv1(x)
        residual = x
        x = F.relu(self.resnet.res_block(x) + residual)
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        out = torch.mean(gru_out, dim=1)
        return self.fc(out)

def get_model(model_name, input_features, num_classes):
    model_name = model_name.lower()
    if model_name in ['proposed', '1d-resnest-bigru', 'resnest-bigru']:
        return IoT_NIDS_Net(input_features, num_classes)
    elif model_name in ['dual-engine', 'dual_engine', 'dual']:
        return DualEngineIoT_NIDS(input_features, num_classes)
    elif model_name == 'cnn':
        return CNN1D(input_features, num_classes)
    elif model_name == 'resnet':
        return ResNet1D(input_features, num_classes)
    elif model_name == 'resnest':
        return IoT_NIDS_Net(input_features, num_classes)
    elif model_name in ['resnet-gru', 'resnet-bigru']:
        return ResNetGRU1D(input_features, num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: proposed, dual-engine, cnn, resnet, resnest, resnet-gru")


# =====================================================================
# 3. DUAL-ENGINE NOVEL EXTENSION (ENGINE 1 + ENGINE 2)
# =====================================================================

class Engine1_ZeroDayAutoencoder(nn.Module):
    """
    ENGINE 1: Unsupervised Deep Reconstruction Autoencoder (Zero-Day Guard).
    Trained ONLY on Benign Normal traffic.
    Computes Reconstruction Loss L_recon = || X - Decoded(X) ||^2.
    If L_recon > tau (where tau = mean_benign + 3 * std_benign), flags as Zero-Day Threat!
    """
    def __init__(self, input_features, latent_dim=32):
        super(Engine1_ZeroDayAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_features),
            nn.Sigmoid()
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
    COMPLETE DUAL-ENGINE HYBRID ARCHITECTURE:
    - Engine 1: Unsupervised Zero-Day Guard (Reconstruction Autoencoder)
    - Engine 2: Supervised Known Attack Classifier (1D ResNeSt + BiGRU)
    - Dual Gates: Reconstruction Threshold (tau = 0.0191) & Confidence Threshold (theta = 0.85)
    """
    def __init__(self, input_features, num_classes, tau_threshold=0.0191, theta_threshold=0.85):
        super(DualEngineIoT_NIDS, self).__init__()
        self.engine1_zero_day_guard = Engine1_ZeroDayAutoencoder(input_features)
        self.engine2_classifier = IoT_NIDS_Net(input_features, num_classes)
        self.tau_threshold = tau_threshold
        self.theta_threshold = theta_threshold

    def forward(self, x):
        # Engine 1 Anomaly Check (Reconstruction Loss)
        recon_loss, latent = self.engine1_zero_day_guard.compute_anomaly_score(x)
        
        # Engine 2 Multi-Class Classifier & Softmax Confidence
        class_logits = self.engine2_classifier(x)
        probs = F.softmax(class_logits, dim=-1)
        p_max, _ = torch.max(probs, dim=-1)
        
        # Dual Gate Decision Rule:
        # Gate 1: L_recon > tau (Anomalous / Non-benign traffic)
        # Gate 2: P_max < theta (Low prediction confidence / Novel attack variant)
        is_anomaly = recon_loss > self.tau_threshold
        is_low_confidence = p_max < self.theta_threshold
        
        # Zero-Day Threat is flagged when traffic is anomalous AND Engine 2 is uncertain
        is_zero_day = is_anomaly & is_low_confidence

        return class_logits, is_zero_day, recon_loss, p_max

    def predict_verdict(self, x):
        """
        Returns 3-way Security Verdict for each input sample:
        - 0: BENIGN NORMAL TRAFFIC (L_recon <= tau)
        - 1: KNOWN ATTACK CLASS (L_recon > tau AND P_max >= theta)
        - 2: ZERO-DAY THREAT ALERT (L_recon > tau AND P_max < theta)
        """
        class_logits, is_zero_day, recon_loss, p_max = self.forward(x)
        _, preds = torch.max(class_logits, dim=-1)
        
        verdicts = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        is_anomaly = recon_loss > self.tau_threshold
        
        # Known Attack: Anomaly + High Confidence
        verdicts[is_anomaly & (p_max >= self.theta_threshold)] = 1
        
        # Zero-Day Threat: Anomaly + Low Confidence
        verdicts[is_zero_day] = 2
        
        return verdicts, preds, p_max, recon_loss


