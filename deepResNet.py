import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Type, Union, List
from deepmoe_utils import MoELayer, ShallowEmbeddingNetwork, MultiHeadedSparseGatingNetwork
# DeepMoE model

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MoEBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        wide: bool = False,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        if wide:
            in_channels = in_channels * 2
            out_channels = out_channels * 2

        self.conv1 = MoELayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MoELayer(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

        self.gate1 = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)
        self.gate2 = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)
    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        identity = x
        
        gate_values_1 = self.gate1(embedding)
        out = self.conv1(x, gate_values_1)
        out = self.bn1(out)
        out = self.relu(out)

        gate_values_2 = self.gate2(embedding)
        out = self.conv2(out, gate_values_2)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, [gate_values_1, gate_values_2]


"""
These bottlenecks are slightly different from the paper's bottlenecks because it simply doesn't make
any sense to have 3 1x1 convolutions in a row. Thus, I followed the bottleneck structure of the original
ResNet paper and applied the MoE layers as according to the paper.
"""

class MoEBottleneckA(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        wide: bool = False,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        mid_channels = int(out_channels * (base_width / 64.)) * groups

        if wide:
            in_channels = in_channels * 2
            mid_channels = mid_channels * 2
            out_channels = out_channels * 2

        self.conv1 = MoELayer(in_channels, mid_channels, kernel_size=1)
        self.bn1 = norm_layer(mid_channels)
        self.conv2 = MoELayer(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, dilation=dilation)
        self.bn2 = norm_layer(mid_channels)
        self.conv3 = conv1x1(mid_channels, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.gate1 = MultiHeadedSparseGatingNetwork(emb_dim, mid_channels)
        self.gate2 = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        identity = x

        gate_values_1 = self.gate1(embeddings)
        out = self.conv1(x, gate_values_1)
        out = self.bn1(out)
        out = self.relu(out)

        gate_values_2 = self.gate2(embeddings)
        out = self.conv2(out, gate_values_2)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, [gate_values_1, gate_values_2]

class MoEBottleneckB(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        wide: bool = False,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        mid_channels = int(out_channels * (base_width / 64.)) * groups

        if wide:
            in_channels = in_channels * 2
            mid_channels = mid_channels * 2
            out_channels = out_channels * 2

        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = norm_layer(mid_channels)
        self.conv2 = MoELayer(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, dilation=dilation)
        self.bn2 = norm_layer(mid_channels)
        self.conv3 = conv1x1(mid_channels, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.gate = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        gate_values = self.gate(embeddings)
        out = self.conv2(out, gate_values)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, [gate_values]


class ResNetMoe(nn.Module):
    def __init__(
        self,
        block: Type[Union[MoEBasicBlock, MoEBottleneckA, MoEBottleneckB]],
        layers: List[int],
        dim: int = 512,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        wide: bool = False,
        cifar: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element list, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.dim = dim

        self.embedding = ShallowEmbeddingNetwork(dim, 3, cifar)

        wide_multiplier = 2 if wide else 1
        self.conv1 = nn.Conv2d(3, self.in_channels * wide_multiplier, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels * wide_multiplier)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], wide=wide)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], wide=wide)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], wide=wide)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], wide=wide)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if wide:
            self.fc = nn.Linear(512 * 2 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.embedding_classifier = nn.Linear(dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MoEBottleneckA) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, MoEBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[MoEBasicBlock, MoEBottleneckA, MoEBottleneckB]],
        out_channels: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        wide: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        wide_multiplier = 2 if wide else 1
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels * wide_multiplier, out_channels * block.expansion * wide_multiplier, stride),
                norm_layer(out_channels * wide_multiplier * block.expansion),
            )

        layers = nn.ModuleList()
        layers.append(
            block(
                self.in_channels, out_channels, self.dim, wide, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    self.dim,
                    wide,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )


        return layers
    def forward(self, x: torch.Tensor, predict=False) -> torch.Tensor:
        embedding = self.embedding(x)
        emb_y_hat = self.embedding_classifier(embedding)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        gates = []
        for layer in self.layer1:
            x, gate = layer(x, embedding)
            gates.extend(gate)
        for layer in self.layer2:
            x, gate = layer(x, embedding)
            gates.extend(gate)
        for layer in self.layer3:
            x, gate = layer(x, embedding)
            gates.extend(gate)
        for layer in self.layer4:
            x, gate = layer(x, embedding)
            gates.extend(gate)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if predict:
            return x
        return x, gates, emb_y_hat
    

def resnet18_moe(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34_moe(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50_moe_a(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBottleneckA, [3, 4, 6, 3], **kwargs)

def resnet50_moe_b(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBottleneckB, [3, 4, 6, 3], **kwargs)

def resnet101_moe_a(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBottleneckA, [3, 4, 23, 3], **kwargs)

def resnet101_moe_b(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBottleneckB, [3, 4, 23, 3], **kwargs)

def resnet152_moe_a(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBottleneckA, [3, 8, 36, 3], **kwargs)

def resnet152_moe_b(**kwargs: Any) -> ResNetMoe:
    return ResNetMoe(MoEBottleneckB, [3, 8, 36, 3], **kwargs)
