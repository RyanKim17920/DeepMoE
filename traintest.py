from functools import partial
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from deepmoe_utils import deepmoe_loss
from deepResNet import resnet18_moe, resnet34_moe, resnet50_moe_a, resnet50_moe_b, resnet101_moe_a, resnet101_moe_b, resnet152_moe_a, resnet152_moe_b
from ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from deepVGG import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, vgg11_moe, vgg11_bn_moe, vgg13_moe, vgg13_bn_moe, vgg16_moe, vgg16_bn_moe, vgg19_moe, vgg19_bn_moe
import argparse

def initialize_model(model_class, num_classes, device):
    return model_class(num_classes=num_classes).to(device)

def initialize_optimizer(model, lr=0.001, optimizer_type="adam", weight_decay=0.0, **kwargs):

    # Select optimizer with extra params
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=kwargs.get("momentum", 0.9), weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=kwargs.get("momentum", 0.9), weight_decay=weight_decay, eps=kwargs.get("eps", 1e-8))
    elif optimizer_type == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer

def initialize_scheduler(optimizer, milestones=None, gamma=0.1):
    # If no milestones are specified, return a constant LR scheduler
    if milestones is None or not milestones:
        return MultiStepLR(optimizer, milestones=[1e10], gamma=1.0)  # Never steps
    
    # Return a MultiStepLR scheduler if milestones are provided
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

def update_criterion(epoch, num_epochs, lambda_val, mu, model=None, num_frozen_epochs=5):
    if epoch >= num_epochs - num_frozen_epochs:
        # Freeze the embedding layer
        if model is not None and hasattr(model, 'embedding'):
            for param in model.embedding.parameters():
                param.requires_grad = False
        return deepmoe_loss(lambda_val=0.0, mu=0.0)
    
    # Unfreeze the embedding layer for earlier epochs if needed
    if model is not None and hasattr(model, 'embedding'):
        for param in model.embedding.parameters():
            param.requires_grad = True
            
    return deepmoe_loss(lambda_val=lambda_val, mu=mu)


scaler = GradScaler()

def train(model, device, train_loader, optimizer, criterion, print_every, moe=False, accumulation_steps=1):
    model.train()
    total_loss, correct = 0, 0
    optimizer.zero_grad(set_to_none=True)  # Zero gradients
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Mixed precision training
        with autocast():
            if not moe:
                output = model(data)
                loss = criterion(output, target)
            else:
                output, gates, emb_y_hat = model(data)
                loss = criterion(output, emb_y_hat, target, gates)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        
        # Accumulate gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Tracking metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Print progress if specified
        if print_every is not None and (batch_idx + 1) % print_every == 0:
            print(f'Train Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {total_loss / (batch_idx + 1):.4f}, Accuracy: {correct / ((batch_idx + 1) * train_loader.batch_size):.4f}')
    
    return total_loss / len(train_loader), correct / len(train_loader.dataset)

def validate(model, device, val_loader, criterion, moe=False):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            if not moe:
                output = model(data)
                loss = criterion(output, target)
            else:
                output, gates, emb_y_hat = model(data)
                loss = criterion(output, emb_y_hat, target, gates)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    return total_loss / len(val_loader), correct / len(val_loader.dataset)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, monitor="val_loss", verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.epochs_without_improvement = 0
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, metric):
        score = -metric if self.monitor == "val_loss" else metric

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.epochs_without_improvement} out of {self.patience}")
            if self.epochs_without_improvement >= self.patience:
                self.early_stop = True

def train_and_validate(model, device, train_loader, val_loader, optimizer, criterion, epoch, print_every, is_moe=False, accumulation_steps=1):
    # Training step
    start_time_train = time.time()
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, print_every, is_moe, accumulation_steps)
    train_duration = time.time() - start_time_train

    # Validation step
    start_time_val = time.time()
    val_loss, val_acc = validate(model, device, val_loader, criterion, is_moe)
    val_duration = time.time() - start_time_val

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Training Duration: {train_duration:.2f} seconds | Validation Duration: {val_duration:.2f} seconds")
    
    return train_loss, val_loss, val_acc

def run_training_loop(
    model_class, num_classes, device, num_epochs, frozen_epochs, train_loader, val_loader, 
    lambda_val, mu, gradient_accumulation, lr=0.001, optimizer_type="adam", is_moe=False, 
    print_every=None, milestones=None, gamma = 0.1, early_stop_metric="val_loss", patience=5
):
    # Initialize model, optimizer, criterion, and scheduler
    model = initialize_model(model_class, num_classes, device)
    optimizer = initialize_optimizer(model, lr=lr, optimizer_type=optimizer_type)
    criterion = deepmoe_loss(lambda_val=lambda_val, mu=mu) if is_moe else nn.CrossEntropyLoss()
    scheduler = initialize_scheduler(optimizer, milestones=milestones, gamma=gamma)
    
    # Early stopping setup
    early_stopping = EarlyStopping(patience=patience, monitor=early_stop_metric, verbose=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Update criterion if using mixture of experts (MOE)
        if is_moe:
            criterion = update_criterion(epoch, num_epochs, lambda_val, mu, model, frozen_epochs)

        # Train and validate
        train_loss, val_loss, val_acc = train_and_validate(
            model, device, train_loader, val_loader, optimizer, criterion, epoch, 
            print_every, is_moe, gradient_accumulation
        )

        # Scheduler step
        scheduler.step()

        # Early stopping check based on selected metric
        metric_to_monitor = val_loss if early_stop_metric == "val_loss" else val_acc
        early_stopping(metric_to_monitor)

        # Stop if early stopping criteria are met
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

def get_transforms(dataset_name):
    if dataset_name.lower() == "cifar100" or dataset_name.lower() == "cifar10":
        # CIFAR specific transformations
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10), 
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        # ImageNet-like transformations (for general datasets)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


def main():

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10", None], help="Dataset to train on (cifar10, cifar100, or None)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store the dataset if using external data")

    parser.add_argument("--model", type=str, default="resnet18", help="Model to train")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes in the dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on")

    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train the model")
    parser.add_argument("--milestones", type=int, nargs="+", help="Milestones for the scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma value for the scheduler")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Number of epochs to train at end with frozen embedding layer")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--gradient_accumulation" , type=int, default=1, help="Gradient Accumulation for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamw", "adamax"], help="Optimizer type")
    parser.add_argument("--print_every", type=int, default=None, help="Number of batches between printing training progress")
    parser.add_argument("--early_stop_metric", type=str, default="val_acc", choices=["val_loss", "val_acc"], help="Metric to monitor for early stopping (either 'val_loss' or 'val_acc').")
    parser.add_argument("--patience", type=int, default=1e10, help="Number of epochs with no improvement after which training will be stopped.")


    parser.add_argument("--lambda_val", type=float, default=0.01, help="Lambda value for the deepmoe loss if using a moe model")
    parser.add_argument("--mu", type=float, default=1, help="Mu value for the deepmoe loss if using a moe model")
    parser.add_argument("--wide", action="store_true", help="Whether to use the wide version of the model")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of the embedding network if using a moe model")
    
    parser.add_argument("--train_paper", type=str, choices = ["cifar", "imagenet", None], help="Train the model as in the paper")

    args = parser.parse_args()

    if args.train_paper == "cifar":
        args.lr = 0.1
        args.epochs = 350
        args.milestones = [150, 250]
        args.freeze_epochs = 80
    elif args.train_paper == "imagenet":
        args.lr = 0.1
        args.epochs = 210 # Total epochs are not given in paper
        args.milestones = [100, 130, 160, 190] 
        args.freeze_epochs = 30 # Not given in paper



    
    transform = get_transforms(args.dataset)
    
    cifar = (args.dataset.lower() == "cifar100" or args.dataset.lower() == "cifar10")
    if args.dataset.lower() == "cifar100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        args.num_classes = 100
    elif args.dataset.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        args.num_classes = 10
    else:
        train_dataset = datasets.ImageFolder(root=f'{args.data_dir}/train', transform=transform)
        val_dataset = datasets.ImageFolder(root=f'{args.data_dir}/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_class = {
        "resnet18": partial(resnet18),
        "resnet34": partial(resnet34),
        "resnet50": partial(resnet50),
        "resnet101": partial(resnet101),
        "resnet152": partial(resnet152),
        "resnet18_moe": partial(resnet18_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "resnet34_moe": partial(resnet34_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "resnet50_moe_a": partial(resnet50_moe_a, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "resnet50_moe_b": partial(resnet50_moe_b, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "resnet101_moe_a": partial(resnet101_moe_a, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "resnet101_moe_b": partial(resnet101_moe_b, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "resnet152_moe_a": partial(resnet152_moe_a, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "resnet152_moe_b": partial(resnet152_moe_b, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg11": partial(vgg11),
        "vgg11_bn": partial(vgg11_bn),
        "vgg13": partial(vgg13),
        "vgg13_bn": partial(vgg13_bn),
        "vgg16": partial(vgg16),
        "vgg16_bn": partial(vgg16_bn),
        "vgg19": partial(vgg19),
        "vgg19_bn": partial(vgg19_bn),
        "vgg11_moe": partial(vgg11_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg11_bn_moe": partial(vgg11_bn_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg13_moe": partial(vgg13_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg13_bn_moe": partial(vgg13_bn_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg16_moe": partial(vgg16_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg16_bn_moe": partial(vgg16_bn_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg19_moe": partial(vgg19_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
        "vgg19_bn_moe": partial(vgg19_bn_moe, wide=args.wide, cifar=cifar, dim=args.embedding_dim),
    }[args.model]

    run_training_loop(model_class, args.num_classes, args.device, args.epochs, args.freeze_epochs,
                      train_loader, val_loader, 
                      args.lambda_val, args.mu, args.gradient_accumulation, args.lr, args.optimizer, 
                      "moe" in args.model, 
                      print_every = args.print_every,
                      milestones = args.milestones,
                      gamma = args.gamma,
                      early_stop_metric = args.early_stop_metric,
                      patience = args.patience)

if __name__ == "__main__":
    main()