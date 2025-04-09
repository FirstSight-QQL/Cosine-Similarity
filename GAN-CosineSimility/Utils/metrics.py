import torch
import torch.nn.functional as F

def calculate_similarity(C, real_pc, fake_pc):
    """Using a comparator to calculate feature similarity"""
    with torch.no_grad():
        real_feat = C(real_pc)
        fake_feat = C(fake_pc)
    return F.cosine_similarity(real_feat, fake_feat).mean().item()