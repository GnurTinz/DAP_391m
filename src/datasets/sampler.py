import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class PKSampler(Sampler):
    """
    BatchSampler that samples P classes and K instances per class to form a batch of size P * K.
    Useful for Contrastive Learning (SupConLoss) to ensure every batch has positive pairs.
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
        
        # Tính số lượng batch xấp xỉ có thể lấy trong 1 epoch
        self.num_batches = max(1, len(self.labels) // self.batch_size)
        
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            # Chọn P danh tính (cho phép trùng lặp nếu số lượng danh tính < P)
            replace_p = len(self.identities) < self.p
            classes = np.random.choice(self.identities, size=self.p, replace=replace_p)
            
            for c in classes:
                indices = self.label_to_indices[c]
                # Chọn K mẫu cho mỗi danh tính
                replace_k = len(indices) < self.k
                sampled_indices = np.random.choice(indices, size=self.k, replace=replace_k)
                batch.extend(sampled_indices)
                
            yield batch
            
    def __len__(self):
        return self.num_batches
