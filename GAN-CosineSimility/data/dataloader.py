import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.preprocess import PointCloudNormalizer

class GeologicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                     if f.endswith('.npy')]
        self.transform = transform
        self.normalizer = PointCloudNormalizer()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc = np.load(self.files[idx])
        if self.transform:
            pc = self.transform(pc)
        pc = self.normalizer.fit(pc).transform(pc)
        return torch.FloatTensor(pc)

def create_dataloader(config):
    dataset = GeologicalDataset(config['data']['root'])
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=True,
        pin_memory=True
    )