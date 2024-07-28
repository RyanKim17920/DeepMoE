# DeepMoE
 Attempting to open source the paper: [Deep Mixture of Experts via Shallow Embedding](https://arxiv.org/abs/1806.01531)

 ### Changes/Assumptions made
 - The original paper uses 3 1x1 convolutions for bottlenecks in the MoE layer which I have replaced the middle layer with a 3x3 kernel size convolution. Having solely 1x1 convolutions in the bottleneck layer fails its purpose of learning complex features while also scaling down the number of parameters. 
 - The paper uses a loss for the embedding layer. I assume that there would be a linear layer after the embedding layer to predict the output to calculate the loss of it.
 - The paper does not acknowledge the size of the embedding dimension. I have assumed it to be 128.

 ### Analysis of the model
 - I have done some preliminary results that doesn't show any significant improvement over the baseline model. The small models simply do worse than the baseline model, which is expected due to the gating of some channels. Additionally, it actually runs slower than the baseline model due to extra calculations.
- The given hyperparameters for the embedding loss and gating loss work well in comparison to other arbitrary choices.
- Training seems to have large amounts of local minima, seen through random canyons in the loss graph.
- Extrapolation to validation data has explosions in loss but accuracy is still manageable.

##### Due to lack of GPU resources, only resnet18 models were trained. However, all models have been tested to run without issues.