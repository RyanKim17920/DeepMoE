# DeepMoE
 Unofficial implementation of the paper: [Deep Mixture of Experts via Shallow Embedding](https://arxiv.org/abs/1806.01531)

# Using the CLI for training

### Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/RyanKim17920/DeepMoE
cd DeepMoE
pip install -r requirements.txt
```

### Getting Started

The main entry point for training is the `main()` function in `traintest.py`. You can run it through the command line, specifying options to customize your training setup. Here’s a basic example:

```bash
python traintest.py --dataset cifar100 --model resnet18 --num_classes 100 --epochs 25
```

### Command Line Arguments

The following are key arguments you can use to configure the training process:

- **Dataset Options**
  - `--dataset`: Choose the dataset to train on. Options are `"cifar10"`, `"cifar100"`, or `None` (for external datasets). Default is `"cifar100"`.
  - `--data_dir`: Specify the directory to store or load the dataset. Default is `"./data"`.

- **Model and Training Options**
  - `--model`: Specify the model architecture. Options include `"resnet18"`, `"resnet50"`, `"vgg16"`, `"vgg19_moe"`, etc. Default is `"resnet18"`.
  - `--num_classes`: Number of classes in the dataset, typically 10 for CIFAR-10 and 100 for CIFAR-100. Default is `100`.
  - `--device`: Device for training. Options are `"cuda"` for GPU or `"cpu"` for CPU. Default is `"cuda"`.
  - `--epochs`: Number of epochs for training. Default is `25`.
  - `--batch_size`: Batch size for training. Default is `64`.
  - `--gradient_accumulation`: Number of steps for gradient accumulation, useful for adjusting effective batch size. Default is `1`.
  - `--num_workers`: Number of worker threads for data loading. Default is `4`.
  - `--print_every`: Interval (in batches) for printing training progress. By default, no progress is printed if not specified.

- **Optimizer and Learning Rate Scheduler Options**
  - `--optimizer`: Optimizer to use. Options include `"adam"`, `"sgd"`, `"rmsprop"`, `"adagrad"`, `"adadelta"`, `"adamw"`, and `"adamax"`. Default is `"adam"`.
  - `--lr`: Learning rate for the optimizer. Default is `0.001`.
  - `--milestones`: List of epochs where the learning rate is adjusted by gamma times for the MultiStepLR scheduler.
  - `--gamma`: Factor by which the learning rate is reduced at each milestone for the MultiStepLR scheduler. Default is `0.1`.
  - `--early_stop_metric`: Metric to use for early stopping. Options are `"val_loss"` or `"val_acc"`. Default is `"val_acc"`.
  - `--patience`: Number of epochs to wait before early stopping if the metric does not improve. Default is `1e10` (essentially negates it).

  - ***Optimizer-specific options***
  - - `--momentum`: Momentum factor for SGD optimizer. Default is `0.9`.
  - - `--weight_decay`: Weight decay (L2 penalty) for the optimizer. Default is `0`.
  - - `--eps`: Epsilon value for the optimizer. Default is `1e-8`.
  - - `--betas`: Betas for the optimizer. Default is `(0.9, 0.999)`.
  - - `--alpha`: Alpha value for the optimizer. Default is `0.99`.
  - - `--lr_decay`: Learning rate decay for the optimizer. Default is `0`.
  - - `--rho`: Rho value for the optimizer. Default is `0.9`.


- **Advanced Options**
  - `--lambda_val`: Lambda value for the `deepmoe_loss` if using a Mixture of Experts (MoE) model. Default is `0.01`.
  - `--freeze_epochs`: Number of epochs to train with a frozen embedding layer, deducted from the total epochs specified. Default is `5`.
  - `--mu`: Mu value for the `deepmoe_loss` if using an MoE model. Default is `1`.
  - `--wide`: Use a wide version of the model if applicable (e.g., ResNet). This is a flag, so no value is needed.
  - `--embedding_dim`: Specify the embedding dimension if using an MoE model. Default is `128`.

- **Paper-Specific Configuration**
  - `--train_paper`: Reproduce settings from the research paper. <span style="color:red"> This is not recommended, refer to [Analysis of the Model](#analysis-of-the-model).</span>. Options are `"cifar"` or `"imagenet"`, with defaults based on the chosen dataset.

### Example Usage

#### Basic Training on CIFAR-100 with ResNet18
To train a `ResNet18` model on CIFAR-100 for 25 epochs:

```bash
python traintest.py --dataset cifar100 --model resnet18 --epochs 25 --lr 0.001 --optimizer adam
```

#### Training a Wide VGG16 Model with Mixture of Experts (MoE) on CIFAR-10
To train a `vgg16_moe` model on CIFAR-10, with specific parameters for MoE loss:

```bash
python traintest.py --dataset cifar10 --model vgg16_moe --epochs 50 --lambda_val 0.01 --mu 1.0 --wide
```

#### Using Paper-Recommended Training Setup
For paper-specific configurations, such as training on CIFAR for 350 epochs:

```bash
python traintest.py --dataset cifar100 --model resnet50 --train_paper cifar
```

### Model Training Process

The training loop includes:
1. Initializing the model, optimizer, learning rate scheduler, and data loaders based on the chosen configurations.
2. Running the training and validation phases, computing and logging loss and accuracy metrics per epoch. Mixed precision training is enabled through `autocast` if supported.
3. Optionally, printing training progress at intervals specified by `print_every`.

This setup allows you to configure and experiment with different models, optimizers, and hyperparameters to tune training for specific datasets and performance targets.



 ## Changes/Assumptions made
 - The original paper uses 3 1x1 convolutions for bottlenecks in the MoE layer which I have replaced the middle layer with a 3x3 kernel size convolution. Having solely 1x1 convolutions in the bottleneck layer fails its purpose of learning complex features while also scaling down the number of parameters. 
 - The paper uses a loss for the embedding layer. I assume that there would be a linear layer after the embedding layer to predict the output to calculate the loss of it.
 - The paper does not acknowledge the size of the embedding dimension nor how the structure of the embedding layers works. I have assumed it to be 128 (this can be modified) with the embedding layers linearly increasing in dimension to the final embedding dimension. This seems to create a large amount of extra parameters that cause the model to run slower than the baseline model.

 ## Analysis of the model
- The given hyperparameters for the embedding loss and gating loss work well in comparison to other arbitrary choices.
- Training seems to have large amounts of local minima, seen through random canyons in the loss graph.
- Extrapolation to validation data has explosions in loss (likely due to the CE loss from routers) but accuracy is still manageable.
- Using the training method from the paper leads to exploding gradients and loss. Lower learning rates are necessary to keep training stability but convergence is very slow. 
- L1 regularization may only decrease activation values instead of zeroing them out, which still uses up extra parameters.
## Further work
- The embedding layer may need to be more complex to learn more complex features.
- Hyperparameters may need to be tuned further.
