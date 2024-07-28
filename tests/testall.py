import torch
from deepmoe_utils import deepmoe_loss
from deepVGG import VGG16, VGGtoDeepMoe
from deepResNet import (
    resnet18_moe,
    resnet34_moe,
    resnet50_moe_a,
    resnet50_moe_b,
    resnet101_moe_a,
    resnet101_moe_b,
    resnet152_moe_a,
    resnet152_moe_b,
)

def test_model(model):
    with torch.no_grad():
        print(model.__class__.__name__)
        input = torch.randn(1, 3, 224, 224).to('cuda')
        target = torch.randint(0, 1000, (1,)).to('cuda').long()
        y_hat, gates, emb_y_hat  = model.to('cuda')(input, predict=False)
        loss = deepmoe_loss()(y_hat, emb_y_hat, target, gates).to('cuda')
        print(f"Loss: {loss.item()}")

models = [
    resnet18_moe(),
    resnet34_moe(),
    resnet50_moe_a(),
    resnet50_moe_b(),
    resnet101_moe_a(),
    resnet101_moe_b(),
    resnet152_moe_a(),
    resnet152_moe_b(),
    resnet18_moe(wide=True),
    resnet34_moe(wide=True),
    resnet50_moe_a(wide=True),
    resnet50_moe_b(wide=True),
    resnet101_moe_a(wide=True),
    resnet101_moe_b(wide=True),
    resnet152_moe_a(wide=True),
    resnet152_moe_b(wide=True),
    VGGtoDeepMoe(VGG16(),wide=False),
    VGGtoDeepMoe(VGG16(),wide=True),
]

for model in models:
    test_model(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print()