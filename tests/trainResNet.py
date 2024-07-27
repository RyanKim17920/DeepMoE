import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from deepResNet import (
    resnet18_moe, resnet34_moe,
    resnet50_moe_a, resnet50_moe_b,
    resnet101_moe_a, resnet101_moe_b,
    resnet152_moe_a, resnet152_moe_b
)
from deepmoe_utils import deepmoe_loss
from deepspeed.ops.adam import DeepSpeedCPUAdam

class ResNetLightningModule(pl.LightningModule):
    def __init__(self, model, is_moe=False):
        super().__init__()
        self.model = model
        self.is_moe = is_moe

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.is_moe:
            y_hat, gates, emb_y_hat = self(x)
            loss = deepmoe_loss(y_hat, emb_y_hat, y, gates)
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, True, True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.is_moe:
            y_hat, gates, emb_y_hat = self(x)
            loss = deepmoe_loss(y_hat, emb_y_hat, y, gates)
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, True, True)
        self.log('val_acc', acc, True, True)

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=0.001)

def get_cifar100_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    return train_loader, test_loader

def train_model(model, is_moe=False, model_name=""):
    train_loader, val_loader = get_cifar100_data()

    lightning_module = ResNetLightningModule(model, is_moe)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )


    # Add ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{model_name}",
        filename="{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stop_callback, checkpoint_callback],
        strategy='deepspeed_stage_3_offload',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    )

    trainer.fit(lightning_module, train_loader, val_loader)

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = ResNetLightningModule.load_from_checkpoint(best_model_path, model=model, is_moe=is_moe)

    return best_model

def compare_models():
    models = {
        'ResNet18': ResNet18(),
        'ResNet34': ResNet34(),
        'ResNet50': ResNet50(),
        'ResNet101': ResNet101(),
        'ResNet152': ResNet152(),
        'ResNet18_MoE': resnet18_moe(wide=False),
        'ResNet34_MoE': resnet34_moe(wide=False),
        'ResNet50_MoE_A': resnet50_moe_a(wide=False),
        'ResNet50_MoE_B': resnet50_moe_b(wide=False),
        'ResNet101_MoE_A': resnet101_moe_a(wide=False),
        'ResNet101_MoE_B': resnet101_moe_b(wide=False),
        'ResNet152_MoE_A': resnet152_moe_a(wide=False),
        'ResNet152_MoE_B': resnet152_moe_b(wide=False),
        'ResNet18_MoE_Wide': resnet18_moe(wide=True),
        'ResNet34_MoE_Wide': resnet34_moe(wide=True),
        'ResNet50_MoE_A_Wide': resnet50_moe_a(wide=True),
        'ResNet50_MoE_B_Wide': resnet50_moe_b(wide=True),
        'ResNet101_MoE_A_Wide': resnet101_moe_a(wide=True),
        'ResNet101_MoE_B_Wide': resnet101_moe_b(wide=True),
        'ResNet152_MoE_A_Wide': resnet152_moe_a(wide=True),
        'ResNet152_MoE_B_Wide': resnet152_moe_b(wide=True),
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        is_moe = '_MoE' in name
        trained_model = train_model(model, is_moe, name)
        results[name] = {
            'val_loss': trained_model.trainer.callback_metrics['val_loss'].item(),
            'val_acc': trained_model.trainer.callback_metrics['val_acc'].item(),
            'best_model_path': trained_model.trainer.checkpoint_callback.best_model_path
        }

    print("\nComparison Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Validation Loss: {metrics['val_loss']:.4f}")
        print(f"  Validation Accuracy: {metrics['val_acc']:.4f}")
        print(f"  Best Model Path: {metrics['best_model_path']}")
        print()

if __name__ == '__main__':
    compare_models() 