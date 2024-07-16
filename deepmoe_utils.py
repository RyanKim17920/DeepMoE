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

class MoELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.out_channels = out_channels
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size, stride, padding) for _ in range(out_channels)])
        
    def forward(self, x, gate_values):
        # Speeding up moe layer by only computing the output for the channels with non-zero gate values
        batch_size, _, height, width = x.shape
        output = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
        
        for i, conv in enumerate(self.conv_layers):
            if gate_values[:, i].sum() > 0:  # If any gate value for this channel is non-zero
                selected_indices = gate_values[:, i] > 0
                selected_x = x[selected_indices]
                if selected_x.size(0) > 0:
                    selected_output = conv(selected_x) * gate_values[selected_indices, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    output[selected_indices, i] = selected_output.squeeze(1)
        
        return output


"""
Blocks follow the same structure as the ones in the paper.
"""

class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = MoELayer(in_channels, in_channels, 3, padding=1)
        self.conv2 = MoELayer(in_channels, in_channels, 3, padding=1)
    def forward(self, x, gate_values):
        x_in = x
        x = self.conv1(x, gate_values)
        x = F.relu(x)
        x = self.conv2(x, gate_values)
        x = F.relu(x)
        return x + x_in
    
class BottleneckA(nn.Module):
    def __init__(self, in_channels, k = 4):
        super().__init__()
        self.conv1 = MoELayer(in_channels, in_channels / k, 1)
        self.conv2 = MoELayer(in_channels / k, in_channels / k, 1)
        self.conv3 = nn.Conv2d(in_channels / k, in_channels, 1)
    def forward(self, x, gate_values):
        x_in = x
        x = self.conv1(x, gate_values)
        x = F.relu(x)
        x = self.conv2(x, gate_values)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x + x_in

class BottleneckB(nn.Module):
    def __init__(self, in_channels, k = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels / k, 1)
        self.conv2 = MoELayer(in_channels / k, in_channels / k, 1)
        self.conv3 = nn.Conv2d(in_channels / k, in_channels, 1)
    def forward(self, x, gate_values):
        x_in = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x, gate_values)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x + x_in

""""Loss function as described in the paper."""

def deepmoe_loss(outputs, embedding_outputs, targets, gates, lambda_val=0.001, mu=1.0):
    base_loss = F.cross_entropy(outputs, targets)
    gate_loss = sum(torch.linalg.norm(g, 1) for g in gates)
    embedding_loss = F.cross_entropy(embedding_outputs, targets)
    return base_loss + lambda_val * gate_loss + mu * embedding_loss