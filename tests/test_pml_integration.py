import unittest
import torch
import torch.nn.functional as F
from src.losses.custom import get_contrastive_loss
from src.datasets.data_module import PalmDataModule

class TestPMLIntegration(unittest.TestCase):
    
    def test_arcface_wrapper(self):
        """
        Verify that ArcFaceWrapper correctly computes loss and exposes last_logits.
        """
        config = {
            'contrastive_type': 'arcface',
            'arcface': {
                'num_classes': 10,
                'embedding_size': 32,
                's': 30.0,
                'm': 0.5
            }
        }
        
        loss_fn = get_contrastive_loss(config)
        features = torch.randn(16, 32)
        labels = torch.randint(0, 10, (16,))
        
        # Forward pass
        loss = loss_fn(features, labels)
        
        # Check if loss is scalar and requires grad
        self.assertTrue(loss.dim() == 0)
        self.assertTrue(loss.requires_grad)
        
        # Check if last_logits are populated and have the correct shape [batch_size, num_classes]
        self.assertTrue(hasattr(loss_fn, 'last_logits'))
        self.assertEqual(loss_fn.last_logits.shape, (16, 10))
        
    def test_supcon_wrapper(self):
        """
        Verify that SupCon wrapper correctly normalizes and computes loss.
        """
        config = {
            'contrastive_type': 'supcon',
            'temperature': 0.1
        }
        loss_fn = get_contrastive_loss(config)
        
        features = torch.randn(8, 32, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]) # Paired labels
        
        loss = loss_fn(features, labels)
        self.assertTrue(loss.dim() == 0)
        self.assertTrue(loss.requires_grad)

    def test_mperclass_sampler_via_datamodule(self):
        """
        Verify that PalmDataModule correctly creates MPerClassSampler when configured.
        """
        # Create a mock config with a tiny subset sizes for testing
        config = {
            'dataset': {
                'name': 'MNISTDataset',
                'data_dir': 'data/MNIST',
                'split_mode': 'ratio',
                'train_ratio': 0.8
            },
            'training': {
                'batch_size': 8,
                'use_sampler': True,
                'sampler_type': 'pk_sampler',
                'sampler_k': 2, # m = 2 in PML
                'num_workers': 0
            }
        }
        
        dm = PalmDataModule(config)
        
        # Setup using a tiny dataset (using random MNIST for speed)
        # Assuming MNISTDataset factory doesn't download instantly, we patch the dataset
        from torch.utils.data import TensorDataset
        import numpy as np
        
        # Mock dataset with 20 samples, 4 classes (0,1,2,3), 5 samples each
        X = torch.randn(20, 1, 28, 28)
        Y = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3])
        mock_dataset = TensorDataset(X, Y)
        # Mock get_labels for the sampler
        mock_dataset.get_labels = lambda: Y.tolist()
        
        dm.train_dataset = mock_dataset
        
        # Try to get dataloader
        loader = dm.train_dataloader()
        
        # Check if the sampler is MPerClassSampler
        from pytorch_metric_learning.samplers import MPerClassSampler
        self.assertIsInstance(loader.sampler, MPerClassSampler)
        
        # Draw one batch and verify that there are exactly k=2 samples per class in the batch
        batch_X, batch_Y = next(iter(loader))
        self.assertEqual(len(batch_Y), 8) # batch_size
        
        counts = np.bincount(batch_Y.numpy())
        for count in counts:
            if count > 0:
                self.assertEqual(count, 2) # Every class present must have exactly k=2 samples

if __name__ == '__main__':
    unittest.main()
