import unittest
import torch
from src.engine.represent import optimize_r_from_latent
from src.models.verifier import TestTimeVerifier

class TestRepresentOptimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.latent_dim = 256
        self.num_samples = 128
        self.steps = 5

    def test_optimize_r_from_latent_basic(self):
        """Test initial finding of r and training verifier"""
        mu_q = torch.randn(1, self.latent_dim)
        logvar_q = torch.randn(1, self.latent_dim)
        
        # Test basic optimization
        r_c, verifier, z_pos, z_neg = optimize_r_from_latent(
            mu_q, logvar_q, self.device, 
            verifier=None, 
            freeze_net=False, 
            num_samples=self.num_samples, 
            max_steps=self.steps, 
            verbose=False
        )
        
        self.assertEqual(r_c.shape, (1, self.latent_dim))
        self.assertTrue(isinstance(verifier, TestTimeVerifier))
        self.assertEqual(z_pos.shape, (self.num_samples, self.latent_dim))

    def test_optimize_r_from_latent_freeze_net(self):
        """Test freezing verifier for subsequent persons"""
        mu_q = torch.randn(1, self.latent_dim)
        logvar_q = torch.randn(1, self.latent_dim)
        
        # Pre-initialize a verifier
        global_verifier = TestTimeVerifier(self.latent_dim, hidden_dim=128).to(self.device)
        # Store original weights for comparison
        original_weight = global_verifier.net[0].weight.clone()
        
        # Optimize with freeze_net=True
        r_c, verifier, z_pos, z_neg = optimize_r_from_latent(
            mu_q, logvar_q, self.device, 
            verifier=global_verifier, 
            freeze_net=True, 
            num_samples=self.num_samples, 
            max_steps=self.steps, 
            verbose=False
        )
        
        # Weights should not have changed
        self.assertTrue(torch.equal(global_verifier.net[0].weight, original_weight))
        self.assertEqual(r_c.shape, (1, self.latent_dim))
        
    def test_verification_mechanism(self):
        """Test that z_pos and r_c can be successfully passed through verifier"""
        mu_q = torch.randn(1, self.latent_dim)
        logvar_q = torch.randn(1, self.latent_dim)
        
        r_c, verifier, z_pos, z_neg = optimize_r_from_latent(
            mu_q, logvar_q, self.device, 
            verifier=None, 
            freeze_net=False, 
            num_samples=self.num_samples, 
            max_steps=self.steps, 
            verbose=False
        )
        
        # Pass z_pos and r_c back into verifier
        with torch.no_grad():
            logits = verifier(z_pos, r_c)
            score_prob = torch.sigmoid(logits).mean().item()
            
        self.assertTrue(0.0 <= score_prob <= 1.0)
        self.assertEqual(logits.shape, (self.num_samples, 1))

if __name__ == '__main__':
    unittest.main()
