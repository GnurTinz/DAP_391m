import torch
import torch.nn as nn
import numpy as np

class PalmNetBackbone(nn.Module):
    """
    End-to-end trained CNN inspired by PalmNet (Gabor-PCA Convolutional Networks).
    Replaces the handcrafted Binarization & Histogram pooling with standard 
    differentiable layers (ReLU, MaxPool, AdaptiveAvgPool) for end-to-end backprop.
    """
    def __init__(self, in_channels=3, use_gabor_init=True, k1=15, k2=15):
        super().__init__()
        self.use_gabor_init = use_gabor_init
        
        # Layer 1: Simulated Gabor filters
        # Kernel size 15x15 to capture large line patterns like in the paper
        self.conv1 = nn.Conv2d(in_channels, k1, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm2d(k1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2: Simulated PCA/Gabor filters
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(k2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Layer 3: Extra projection layer to reach target embedding depth
        self.conv3 = nn.Conv2d(k2, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Spatial Pyramid Pooling equivalent (Adaptive pooling to 1x1 spatial)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.use_gabor_init:
            self._init_gabor()
            
    def _init_gabor(self):
        """
        Initialize the first convolutional layer with mathematical Gabor filters
        according to the parameters defined in the PalmNet paper.
        """
        import cv2
        weights = torch.zeros_like(self.conv1.weight.data)
        
        # Parameters inspired by PalmNet paper Table IV
        sigma = 5.6179
        lambd = 1.0 / 0.11 # wavelength
        gamma = 1.0        # aspect ratio
        psi = 0            # phase offset
        
        for i in range(weights.size(0)):
            # Distribute orientations evenly from 0 to Pi
            theta = i * (np.pi / weights.size(0))
            # Generate 15x15 Gabor kernel
            kernel = cv2.getGaborKernel((15, 15), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            kernel_tensor = torch.from_numpy(kernel)
            
            # Replicate the 2D Gabor kernel across all input channels
            for c in range(weights.size(1)):
                weights[i, c] = kernel_tensor
                
        self.conv1.weight.data = weights

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        return torch.flatten(x, 1)
