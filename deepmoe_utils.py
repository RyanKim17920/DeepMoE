import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ShallowEmbeddingNetwork(nn.Module):
    def __init__(self, dim, input_channels, cifar=False):
        super().__init__()
        """
        As stated by the paper:
        The shallow embedding network maps the raw
        input image into a latent mixture weights to be fed
        into the multi-headed sparse gating network. To
        reduce the computational overhead of the embedding
        network, we use a 4-layer (for CIFAR) or 5-layer (for
        ImageNet) convolutional network with 3-by-3 filters
        with stride 2 (roughly 2% of the computation of the
        base models).

        However, there is no mention of the number of channels so the ones used here are arbitrary. 
        These don't seem like 2% of the computation of the base models however.
        """

        if cifar:
            dim_ = math.floor(dim / 4)
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, dim_, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(dim_, 2 * dim_, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(2 * dim_, 3 * dim_, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(3 * dim_, dim, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            dim_ = math.floor(dim / 5)
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, dim_, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(dim_, 2 * dim_, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(2 * dim_, 3 * dim_, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(3 * dim_, 4 * dim_, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(4 * dim_, dim, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        
    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class MultiHeadedSparseGatingNetwork(nn.Module):
    def __init__(self, embedding_dim, num_experts):
        # Learned parameters Wgl to produce gates by multiplying with embedding output
        # No bias term as a result
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_experts, bias = False)
        
    def forward(self, e):
        return F.relu(self.fc(e))

#"""
class MoELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                  stride=stride, padding=padding, dilation=dilation)
        self.out_channels = out_channels  # Store output channels for reference
        
    def forward(self, x, gate_values):
        gate_values = gate_values.view(gate_values.size(0), self.out_channels, 1, 1)
        output = self.conv_layer(x)
        gated_output = output * gate_values
        return gated_output

""""Loss function as described in the paper."""

class deepmoe_loss(nn.Module):
    def __init__(self, lambda_val=0.001, mu=1.0):
        super(deepmoe_loss, self).__init__()
        self.lambda_val = lambda_val
        self.mu = mu

    def forward(self, outputs, embedding_outputs, targets, gates):
        # Cross-entropy loss for the main outputs
        base_loss = F.cross_entropy(outputs, targets)
        
        # L1 norm of the gate scores
        gate_loss = sum(torch.linalg.norm(g, 1) for g in gates)
        
        # Cross-entropy loss for embedding outputs
        embedding_loss = F.cross_entropy(embedding_outputs, targets)
        
        # Total loss
        total_loss = base_loss + self.lambda_val * gate_loss + self.mu * embedding_loss
        return total_loss