import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """

        if cifar:
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(128, dim, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 192, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(192, dim, 3, stride=2, padding=1),
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
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        
    def forward(self, x, gate_values):
        gate_values = gate_values.to(x.dtype)
        batch_size = x.size(0)
        gate_values = gate_values.view(batch_size, -1, 1, 1)
        output = self.conv_layer(x)
        output = output * gate_values
        return output

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