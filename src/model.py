import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. SPLIT-ATTENTION & RESNEST 1D (TRUE RADIX FEATURE BRANCHES)
# =====================================================================

class SplitAttention1D(nn.Module):
    """
    True Split-Attention 1D:
    Splits input into `radix` channel splits, computes contextual attention 
    over the pooled sum, applies radix-softmax, and aggregates weighted branches:
        V = sum_{i=1}^radix (a_i * U_i)
    """
    def __init__(self, in_channels, channels, radix=2, reduction_factor=4):
        super(SplitAttention1D, self).__init__()
        self.radix = radix
        self.channels = channels
        inter_channels = max(16, (channels * radix) // reduction_factor)
        
        self.fc1 = nn.Linear(channels, inter_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(inter_channels, channels * radix)

    def forward(self, x):
        # Input shape: (B, channels * radix, L)
        batch_size = x.size(0)
        
        # Split along channel dimension into radix branches: list of (B, channels, L)
        splits = torch.split(x, self.channels, dim=1)
        
        # Element-wise summation across cardinal splits: U_tilde = sum(U_i)
        u_tilde = sum(splits)  # Shape: (B, channels, L)
        
        # Global context vector via 1D average pooling
        gap = u_tilde.mean(dim=-1)  # Shape: (B, channels)
        
        # Dense channel attention
        att = self.fc1(gap)
        att = self.bn1(att)
        att = self.relu(att)
        att = self.fc2(att)  # Shape: (B, channels * radix)
        
        # Reshape to (B, radix, channels) and apply softmax over radix dimension
        att = att.view(batch_size, self.radix, self.channels)
        att = F.softmax(att, dim=1)  # Shape: (B, radix, channels)
        
        # Channel-wise weighted sum across radix branches
        out = 0
        for r, split in enumerate(splits):
            weight_r = att[:, r, :].unsqueeze(-1)  # Shape: (B, channels, 1)
            out = out + (split * weight_r)
            
        return out


class ResNeStBlock1D(nn.Module):
    """
    ResNeSt Residual Unit with Radix Branches & Split-Attention
    """
    def __init__(self, in_channels, out_channels, radix=2, groups=2):
        super(ResNeStBlock1D, self).__init__()
        self.radix = radix
        self.out_channels = out_channels
        mid_channels = out_channels * radix
        
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=groups * radix, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        
        self.split_attention = SplitAttention1D(in_channels=mid_channels, channels=out_channels, radix=radix)
        
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
    Engine 2 Classifier: 1D-ResNeSt with Split-Attention + BiGRU
    """
    def __init__(self, input_features, num_classes, hidden_gru=64, dropout_rate=0.5):
        super(IoT_NIDS_Net, self).__init__()
        
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
        
        self.resblock = ResNeStBlock1D(in_channels=128, out_channels=128, radix=2, groups=2)
        
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
            x = x.unsqueeze(1)
            
        x = self.spatial_extractor(x)
        x = self.resblock(x)
        
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        
        context_vector = torch.mean(gru_out, dim=1)
        out = self.dropout(context_vector)
        return self.fc(out)


# =====================================================================
# 2. BASELINE ARCHITECTURES
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
# 3. DUAL-ENGINE ARCHITECTURE (ENGINE 1 AUTOENCODER + ENGINE 2 GUARD)
# =====================================================================

class Engine1_ZeroDayAutoencoder(nn.Module):
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
        if x.dim() == 3:
            x = x.squeeze(1)
        reconstructed, latent = self.forward(x)
        loss = torch.mean((x - reconstructed) ** 2, dim=1)
        return loss, latent


class DualEngineIoT_NIDS(nn.Module):
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
        
        is_anomaly = recon_loss > self.tau_threshold
        is_low_confidence = p_max < self.theta_threshold
        is_zero_day = is_anomaly | is_low_confidence

        return class_logits, is_zero_day, recon_loss, p_max

    def predict_verdict(self, x, normal_class_idx=0):
        class_logits, is_zero_day, recon_loss, p_max = self.forward(x)
        _, preds = torch.max(class_logits, dim=-1)
        
        # 0: Benign, 1: Known Attack, 2: Zero-Day / Anomaly
        verdicts = torch.full((x.size(0),), 2, dtype=torch.long, device=x.device)
        is_normal_recon = recon_loss <= self.tau_threshold
        is_high_conf = p_max >= self.theta_threshold
        
        verdicts[is_normal_recon & is_high_conf & (preds == normal_class_idx)] = 0
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