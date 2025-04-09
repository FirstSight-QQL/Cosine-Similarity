import argparse
import yaml
import torch
from tqdm import tqdm
import numpy as np
from data.dataloader import create_dataloader
from models.generator import ClusterGenerator
from Utils.metrics import calculate_similarity
from Utils.visualization import visualize_comparison


def load_model(config, checkpoint_path):
    model = ClusterGenerator(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['G'])
    return model.to(config['train']['device'])


def evaluate(config, model, dataloader, device):
    model.eval()
    cd_values = []
    fs_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            real_pc = batch.to(device)
            fake_pc = model(real_pc)

            # Calculate indicators
            fs = calculate_similarity(real_pc, fake_pc).cpu().item()

            fs_values.append(fs)

            # Visualize the first sample
            if len(cd_values) == 1:
                orig_np = real_pc[0].cpu().numpy()
                simp_np = fake_pc[0].cpu().numpy()
                visualize_comparison(orig_np, simp_np)

    return {
        'Feature_Similarity': np.mean(fs_values)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', default='./datasets/test')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Covering the test data path
    config['data']['path'] = args.data_path

    device = torch.device(config['train']['device'])
    model = load_model(config, args.checkpoint)
    loader = create_dataloader(config['data'])

    metrics = evaluate(config, model, loader, device)
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")