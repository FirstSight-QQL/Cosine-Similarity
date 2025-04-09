import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.cluster import parallel_dbscan



class DynamicMultiheadAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, labels):
        unique_labels = torch.unique(labels[labels != -1])
        outputs = torch.zeros_like(x)

        for label in unique_labels:
            mask = (labels == label)
            group = x[mask]
            q = self.q_proj(group)
            k = self.k_proj(group)
            v = self.v_proj(group)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            outputs[mask] = attn_out

        return self.out_proj(outputs)


class ClusterGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = DynamicMultiheadAttention(config['model']['input_dim'])
        self.mlp = nn.Sequential(
            nn.Linear(3, config['model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['model']['hidden_dim'], 3))

    def cluster_points(self, x):
        return parallel_dbscan(x.cpu().numpy(),
                               eps=self.config['model']['eps'],
                               min_samples=self.config['model']['min_samples'])

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            pc = x[i]
            labels = torch.tensor(self.cluster_points(pc), device=x.device)
            attn_out = self.attention(pc, labels)
            transformed = self.mlp(attn_out)
            outputs.append(transformed[:self.config['model']['K']])
        return torch.stack(outputs)