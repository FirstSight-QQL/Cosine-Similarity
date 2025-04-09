# Geological Point Cloud Decimation with GANs
Implementation of Geological Point Cloud Simplification Algorithm Based on Generative Adversarial Networks and Cosine Similarity
##Functional characteristics
-Dynamic adaptive multi head attention mechanism
-Feature preservation strategy based on DBSCAN clustering
-Hybrid Adversarial Training Framework
-3D point cloud visualization support
-A complete training monitoring and evaluation system
####Data preparation
```bash
Save point cloud data in. npy format
Organize the dataset according to the following structure:
datasets/
├── train/
│   ├── cloud_001.npy
│   └── cloud_002.npy
└── test/
    ├── test_001.npy
    └── test_002.npy
```

Environment configuration
```bash
conda create -n pc-gan python=3.9
conda activate pc-gan
pip install -r requirements.txt
```
Quiky start
```
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py \
    --checkpoint checkpoints/epoch_100.pth \
    --data_path datasets/test

