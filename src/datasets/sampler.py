import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class PKSampler(Sampler):
    """
    BatchSampler that samples P classes and K instances per class to form a batch of size P * K.
    Useful for Contrastive Learning (SupConLoss, TripletLoss).
    """
    def __init__(self, labels, p=16, k=4):
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.labels = np.array(labels)
        
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
            
        self.identities = list(self.label_to_indices.keys())
        self.num_batches = max(1, len(self.labels) // self.batch_size)
        
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            replace_p = len(self.identities) < self.p
            classes = np.random.choice(self.identities, size=self.p, replace=replace_p)
            
            for c in classes:
                indices = self.label_to_indices[c]
                replace_k = len(indices) < self.k
                sampled_indices = np.random.choice(indices, size=self.k, replace=replace_k)
                batch.extend(sampled_indices)
                
            yield batch
            
    def __len__(self):
        return self.num_batches

class WeightedClassSampler(Sampler):
    """
    Standard WeightedRandomSampler approach but wrapped as a BatchSampler.
    Samples batches based on inverse class frequency to handle highly imbalanced datasets.
    """
    def __init__(self, labels, batch_size=32):
        self.batch_size = batch_size
        self.labels = np.array(labels)
        
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        
        self.sample_weights = class_weights[self.labels]
        self.sample_weights /= self.sample_weights.sum()
        
        self.num_batches = max(1, len(self.labels) // self.batch_size)
        
    def __iter__(self):
        for _ in range(self.num_batches):
            # Sample indices based on computed weights
            batch = np.random.choice(
                len(self.labels), 
                size=self.batch_size, 
                replace=True, 
                p=self.sample_weights
            )
            yield batch.tolist()
            
    def __len__(self):
        return self.num_batches

def get_sampler(sampler_type, labels, batch_size, p=None, k=None):
    """
    Factory method để chọn sampler tùy thuộc cấu hình yaml.
    """
    if sampler_type == 'pk_sampler':
        if p is None or k is None:
            k = 4
            p = max(1, batch_size // k)
        return PKSampler(labels, p=p, k=k)
        
    elif sampler_type == 'weighted':
        return WeightedClassSampler(labels, batch_size=batch_size)
        
    else:
        raise ValueError(f"Sampler type {sampler_type} không được hỗ trợ!")
