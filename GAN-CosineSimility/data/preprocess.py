import numpy as np
import torch


class PointCloudNormalizer:
    """Point Cloud Normalization Processor"""

    def __init__(self, min_range=-1, max_range=1):
        self.min_range = min_range
        self.max_range = max_range

    def fit(self, pc):
        self.data_min = pc.min(axis=0)
        self.data_max = pc.max(axis=0)
        return self

    def transform(self, pc):
        pc = (pc - self.data_min) / (self.data_max - self.data_min + 1e-8)
        return pc * (self.max_range - self.min_range) + self.min_range

    def inverse_transform(self, pc):
        pc = (pc - self.min_range) / (self.max_range - self.min_range)
        return pc * (self.data_max - self.data_min) + self.data_min