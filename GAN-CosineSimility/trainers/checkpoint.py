import torch
import os


class CheckpointManager:
    """Model Checkpoint Manager"""

    def __init__(self, save_dir, max_to_keep=5):
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model_dict, epoch):
        filename = os.path.join(self.save_dir, f"epoch_{epoch}.pth")
        torch.save(model_dict, filename)
        self._cleanup_old(epoch)

    def _cleanup_old(self, current_epoch):
        """Keep the latest max_to'keep checkpoints"""
        checkpoints = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith('.pth')],
            key=lambda x: int(x.split('_')[1].split('.')[0]))

        if len(checkpoints) > self.max_to_keep:
            for f in checkpoints[:-self.max_to_keep]:
                os.remove(os.path.join(self.save_dir, f))

    def load_latest(self, device):
        """Load the latest checkpoint"""
        checkpoints = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith('.pth')],
            key=lambda x: int(x.split('_')[1].split('.')[0]))

        if not checkpoints:
            return None

        latest = checkpoints[-1]
        return torch.load(os.path.join(self.save_dir, latest), map_location=device)