import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-9, 1.0))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class ResNetArcFace(nn.Module):
    def __init__(self, num_classes, feature_dim=512, pretrained=True):
        super(ResNetArcFace, self).__init__()
        # Use updated torchvision weights enum
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc = nn.Linear(resnet.fc.in_features, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.arcface = ArcFace(feature_dim, num_classes)
        
    def forward(self, x, labels=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.bn(self.fc(x))
        
        if labels is not None:
            logits = self.arcface(features, labels)
            return features, logits
        return features
