import yaml
import argparse
from data.dataloader import create_dataloader
from trainers.trainer import GeoPCTrainer
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # create directory
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['train']['log_dir'], exist_ok=True)

    # initialization
    dataloader = create_dataloader(config)
    trainer = GeoPCTrainer(config)

    # started training
    trainer.train(dataloader)


if __name__ == '__main__':
    main()