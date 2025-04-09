import os
import torch
from tqdm import tqdm
from losses.hybrid_loss import HybridLoss
from models.discriminator import GeometricDiscriminator
from models.comparator import FeatureComparator
from models.generator import ClusterGenerator
from trainers.checkpoint import CheckpointManager

class GeoPCTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['train']['device'])

        self.G = ClusterGenerator(config).to(self.device)
        self.D = GeometricDiscriminator(config).to(self.device)
        self.C = FeatureComparator(config).to(self.device)

        self.opt_C = torch.optim.Adam(
            self.C.parameters(),
            lr=config['train']['lr'] * 0.1
        )

        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=config['train']['lr'],
            betas=config['train']['betas']
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=config['train']['lr'] * 0.5
        )

        self.logger = HybridLoss(config['train']['log_dir'])
        self.checkpoint_mgr = CheckpointManager(
            config['train']['checkpoint_dir'],
            max_to_keep=config['train']['max_to_keep']
        )
        self.criterion = HybridLoss(config)

    def _train_step(self, real_pc):
        fake_pc = self.G(real_pc)

        self.opt_D.zero_grad()
        d_loss = self.criterion(real_pc, fake_pc, self.D, self.C)
        d_loss.backward()
        self.opt_D.step()


        self.opt_G.zero_grad()
        self.opt_C.zero_grad()
        g_loss = self.criterion(real_pc, fake_pc, self.D, self.C)
        g_loss.backward()
        self.opt_G.step()
        self.opt_C.step()


        self.criterion.update_weights(self.current_iter)
        self.current_iter += 1

        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}

    def train(self, dataloader):
        self.current_iter = 0
        for epoch in range(self.config['train']['epochs']):
            self.G.train()
            self.D.train()
            total_loss = {'d_loss': 0, 'g_loss': 0}

            with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
                for i, batch in enumerate(pbar):
                    real_pc = batch.to(self.device)
                    losses = self._train_step(real_pc)

                    # Record losses
                    total_loss['d_loss'] += losses['d_loss']
                    total_loss['g_loss'] += losses['g_loss']

                    # Update progress bar
                    pbar.set_postfix({
                        'D Loss': losses['d_loss'],
                        'G Loss': losses['g_loss']
                    })

                    # Record logs every 100 steps
                    if i % 100 == 99:
                        avg_dloss = total_loss['d_loss'] / 100
                        avg_gloss = total_loss['g_loss'] / 100
                        self.logger.log_scalar('Loss/D', avg_dloss, epoch * len(dataloader) + i)
                        self.logger.log_scalar('Loss/G', avg_gloss, epoch * len(dataloader) + i)
                        total_loss = {'d_loss': 0, 'g_loss': 0}

            # Saving Checkpoints
            if epoch % 10 == 0:
                self.checkpoint_mgr.save({
                    'G': self.G.state_dict(),
                    'D': self.D.state_dict(),
                    'opt_G': self.opt_G.state_dict(),
                    'opt_D': self.opt_D.state_dict(),
                    'epoch': epoch
                }, epoch)