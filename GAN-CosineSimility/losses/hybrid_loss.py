import torch
import torch.nn.functional as F


class HybridLoss:
    def __init__(self, config):
        self.alpha = config['train']['alpha_init']
        self.beta = config['train']['beta_init']
        self.total_iters = config['train']['epochs'] * config['data']['batch_size']

    def update_weights(self, current_iter):
        """动态调整损失权重"""
        self.alpha = min(0.9, self.alpha + current_iter / self.total_iters)
        self.beta = 1 - self.alpha

    def __call__(self, real_pc, fake_pc, D, C):
        # 判别器损失
        real_pred = D(real_pc)
        fake_pred = D(fake_pc.detach())
        d_loss = -(torch.log(real_pred).mean() + torch.log(1 - fake_pred).mean())

        # 比较器相似度损失
        real_feat = C(real_pc)
        fake_feat = C(fake_pc)
        c_loss = 1 - F.cosine_similarity(real_feat, fake_feat).mean()

        return self.alpha * c_loss + self.beta * d_loss