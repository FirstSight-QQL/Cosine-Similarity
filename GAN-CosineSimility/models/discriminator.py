import torch.nn as nn


class GeometricDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x.mean(dim=1))