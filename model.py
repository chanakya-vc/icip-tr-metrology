import torch
import torch.nn as nn
import torch.optim as optim


class SEMInverseModel_multipixelCNN(nn.Module):
    """
    1D CNN to map a spatial window of SE counts to geometry parameters (h, alpha, w).
    Captures spatial shifts and is inherently more robust to edge spread caused
    by finite spot size and beam-induced damage.
    """

    def __init__(self, window_size=31, output=3, hidden_dim=128):
        super( SEMInverseModel_multipixelCNN, self).__init__()

        # Feature Extractor
        # Input shape: (batch_size, 1 channel, window_size)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Dynamically calculate flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, window_size)
            flat_dim = self.features(dummy_input).view(1, -1).size(1)

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Slight regularization
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.GELU(),
            nn.Linear(int(hidden_dim/2), output)  # Predicts normalized [h, alpha, w]
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.regressor(x)
        return x


class TargetNormalizer:
    """Scales targets to [0, 1] so MSE loss treats h, alpha, and w equally."""

    def __init__(self, target_mins, target_maxs, device='cpu'):
        self.mins = torch.tensor(target_mins, dtype=torch.float32, device=device)
        self.maxs = torch.tensor(target_maxs, dtype=torch.float32, device=device)
        self.range = self.maxs - self.mins

    def normalize(self, targets):
        return (targets - self.mins) / self.range

    def denormalize(self, norm_targets):
        return (norm_targets * self.range) + self.mins


class ResBlock1D(nn.Module):
    """A standard ResNet Basic Block, adapted for 1D sequences."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()  # GELU often performs better than ReLU for continuous physics signals

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (Identity mapping)
        self.shortcut = nn.Sequential()
        # If dimensions change (due to stride or channel increase), we must project the shortcut to match
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # The critical addition that makes it a ResNet!
        out = self.gelu(out)

        return out


class SEMResNet2D(nn.Module):
    def __init__(self, num_targets=3):
        super().__init__()

        # 1. Initial Convolution (Lifts the 1-channel signal into a higher dimensional feature space)
        self.prep = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU()
        )

        # 2. ResNet Layers (Progressively downsample space, upsample channels)
        self.layer1 = ResBlock1D(16, 32, stride=2)  # Halves the spatial dimension (e.g., 31 -> 16)
        self.layer2 = ResBlock1D(32, 64, stride=2)  # Halves again (e.g., 16 -> 8)
        self.layer3 = ResBlock1D(64, 128, stride=2)  # Halves again (e.g., 8 -> 4)

        # 3. Global Pooling (Makes the network robust to slight variations in input window size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. Final Regressor
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_targets)
        )

    def forward(self, x):
        # x shape: (batch, 1, sequence_length)
        x = self.prep(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)  # shape becomes (batch, 128, 1)
        x = torch.flatten(x, 1)  # shape becomes (batch, 128)

        x = self.regressor(x)
        return x


class SEMResNet1D(nn.Module):
    def __init__(self, num_targets=3):
        super().__init__()

        # 1. Initial Convolution (Lifts the 1-channel signal into a higher dimensional feature space)
        self.prep = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU()
        )

        # 2. ResNet Layers (Progressively downsample space, upsample channels)
        self.layer1 = ResBlock1D(16, 32, stride=2)  # Halves the spatial dimension (e.g., 31 -> 16)
        self.layer2 = ResBlock1D(32, 64, stride=2)  # Halves again (e.g., 16 -> 8)
        self.layer3 = ResBlock1D(64, 128, stride=2)  # Halves again (e.g., 8 -> 4)

        # 3. Global Pooling (Makes the network robust to slight variations in input window size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. Final Regressor
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_targets)
        )

    def forward(self, x):
        # x shape: (batch, 1, sequence_length)
        x = self.prep(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)  # shape becomes (batch, 128, 1)
        x = torch.flatten(x, 1)  # shape becomes (batch, 128)

        x = self.regressor(x)
        return x