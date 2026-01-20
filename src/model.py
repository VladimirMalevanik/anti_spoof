import torch
import torch.nn as nn

class MaxFeatureMap(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] % 2 == 0, "Channel dim must be even for MFM"
        c = x.shape[1] // 2
        return torch.max(x[:, :c], x[:, c:])

class LightCNN(nn.Module):
    """
    Вход: [B,1,867,600]
    После слоёв: [B,32,53,37] => 32*53*37
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=(0, 2)),
            MaxFeatureMap(),        # 64->32
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap(),        # 64->32
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MaxFeatureMap(),        # 96->48
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 96, kernel_size=1),
            MaxFeatureMap(),        # 96->48
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MaxFeatureMap(),        # 128->64
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=1),
            MaxFeatureMap(),        # 128->64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            MaxFeatureMap(),        # 64->32
            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap(),        # 64->32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MaxFeatureMap(),        # 64->32
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(32 * 53 * 37, 160)
        self.mfm_fc = MaxFeatureMap()      # 160->80
        self.bn_fc = nn.BatchNorm1d(80)
        self.drop = nn.Dropout(0.75)
        self.fc2 = nn.Linear(80, 2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.mfm_fc(x)
        x = self.bn_fc(x)
        x = self.drop(x)
        return self.fc2(x)
