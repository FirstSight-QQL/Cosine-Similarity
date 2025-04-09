import torch.nn as nn

class FeatureComparator(nn.Module):
    """Feature comparator network"""
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config['model']['feat_dim']
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.feat_dim)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, pc):
        """
        input:
            pc: [B, N, 3]
        output:
            features: [B, feat_dim]
        """
        features = self.mlp(pc)          # [B, N, feat_dim]
        features = features.permute(0,2,1) # [B, feat_dim, N]
        return self.pool(features).squeeze(-1) # [B, feat_dim]