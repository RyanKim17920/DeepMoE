import torch
import torch.nn as nn
from typing import List, Union, Dict, Any, Optional
from deepmoe_utils import ShallowEmbeddingNetwork, MultiHeadedSparseGatingNetwork, MoELayer

class MoeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=False):
        super(MoeBlock, self).__init__()
        self.moe = MoELayer(in_channels, out_channels, kernel_size, stride, padding)
        
        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=False))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x, gate_values):
        x = self.moe(x, gate_values)
        return self.block(x)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, gated: bool = False, wide: bool = False) -> nn.ModuleList:
    layers: nn.ModuleList = nn.ModuleList()
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            v = int(v)
            if wide and i > 0: 
                v *= 2
            if gated:
                layers.append(MoeBlock(in_channels, v, batch_norm=batch_norm))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers.append(nn.Sequential(conv, nn.BatchNorm2d(v), nn.ReLU(inplace=False)))
                else:
                    layers.append(nn.Sequential(conv, nn.ReLU(inplace=True)))
            in_channels = v
    return layers

class VGG(nn.Module):
    def __init__(
        self, 
        features: nn.ModuleList, 
        num_classes: int = 1000, 
        init_weights: bool = True, 
        dropout: float = 0.5,
        gated: bool = False,
        wide: bool = False,
        dim: int = 128,
        cifar: bool = False
    ) -> None:
        super().__init__()
        self.features = features
        self.gated = gated
        self.wide = wide
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Adjust the input features for the classifier based on whether it's a wide model
        classifier_input_features = 512 * (2 if wide else 1) * 7 * 7
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_features, 4096),
            nn.ReLU(False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        
        if gated:
            self.embedding = ShallowEmbeddingNetwork(dim, 3, cifar)
            self.gating = nn.ModuleList([
                MultiHeadedSparseGatingNetwork(dim, layer.moe.out_channels if isinstance(layer, MoeBlock) else layer.moe.out_channels)
                for layer in self.features if not isinstance(layer, nn.MaxPool2d)
            ])
            self.embedding_classifier = nn.Linear(dim, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            emb = self.embedding(x)
            gates = []
            gate_index = 0  
    
            for layer in self.features:
                if isinstance(layer, MoeBlock):
                    gate_value = self.gating[gate_index](emb)
                    gates.append(gate_value)
                    x = layer(x, gate_value)
                    gate_index += 1 
                else:
                    x = layer(x)
    
        else:
            for layer in self.features:
                x = layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        if self.gated:
            emb_class = self.embedding_classifier(emb)
            return x, gates, emb_class
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def _vgg(cfg: str, batch_norm: bool, gated: bool = False, wide: bool = False, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, gated=gated, wide=wide), gated=gated, wide=wide, **kwargs)
    return model

def vgg11_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("A", False, True, wide, **kwargs)

def vgg11_bn_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("A", True, True, wide, **kwargs)

def vgg13_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("B", False, True, wide, **kwargs)

def vgg13_bn_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("B", True, True, wide, **kwargs)

def vgg16_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("D", False, True, wide, **kwargs)

def vgg16_bn_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("D", True, True, wide, **kwargs)

def vgg19_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("E", False, True, wide, **kwargs)

def vgg19_bn_moe(wide: bool = False, **kwargs: Any) -> VGG:
    return _vgg("E", True, True, wide, **kwargs)



def vgg11(**kwargs: Any) -> VGG:
    return _vgg("A", False, **kwargs)

def vgg11_bn(**kwargs: Any) -> VGG:
    return _vgg("A", True, **kwargs)

def vgg13(**kwargs: Any) -> VGG:
    return _vgg("B", False, **kwargs)

def vgg13_bn(**kwargs: Any) -> VGG:
    return _vgg("B", True, **kwargs)

def vgg16(**kwargs: Any) -> VGG:
    return _vgg("D", False, **kwargs)

def vgg16_bn(**kwargs: Any) -> VGG:
    return _vgg("D", True, **kwargs)

def vgg19(**kwargs: Any) -> VGG:
    return _vgg("E", False, **kwargs)

def vgg19_bn(**kwargs: Any) -> VGG:
    return _vgg("E", True, **kwargs)

