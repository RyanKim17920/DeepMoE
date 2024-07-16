import torch.nn as nn
import torch.nn.functional as F
from deepmoe_utils import ShallowEmbeddingNetwork, MultiHeadedSparseGatingNetwork, MoELayer

import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class VGG16(nn.Module):
    def __init__(self, num_classes=10, channels = 3):
        # Normal VGG16 architecture
        super(VGG16, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.features = nn.Sequential(
            ConvBlock(channels, 64),
            ConvBlock(64, 64, pool=True),
            ConvBlock(64, 128),
            ConvBlock(128, 128, pool=True),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256, pool=True),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512, pool=True),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512, pool=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


"""Turning VGG16 into a DeepMoE model"""

class MoeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(MoeBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.moe = MoELayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        layers = [
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x, gate_values):
        x = self.moe(x, gate_values)
        return self.block(x)

class VGGtoDeepMoe(nn.Module):
    def __init__(self, vgg_model, dim = 512, wide = False):
        # Normal VGG16 architecture
        super(VGGtoDeepMoe, self).__init__()
        self.embedding = ShallowEmbeddingNetwork(dim, vgg_model.channels)
        if wide:
            # Make sure the first layer keeps same input channels
            self.features = nn.ModuleList([MoeBlock(vgg_model.channels, vgg_model.features[0].out_channels * 2)])
            # Double the channels for the rest of the layers
            self.features.extend([MoeBlock(i.in_channels * 2, i.out_channels * 2) for i in vgg_model.features[1:]])
        else:
            self.features = nn.ModuleList([
                MoeBlock(i.in_channels, i.out_channels) for i in vgg_model.features
            ])
        self.gating = nn.ModuleList([
            MultiHeadedSparseGatingNetwork(dim, i.out_channels) for i in self.features
        ])
        self.classifier = vgg_model.classifier
    def forward(self, x):
        emb = self.embedding(x)
        for (f, g) in zip(self.features, self.gating):
            x = f(x, g(emb))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
