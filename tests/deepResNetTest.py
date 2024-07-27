import torch
from ResNet import resnet18
from deepResNet import resnet18_moe
from deepmoe_utils import deepmoe_loss
"""


resnet_m = resnet18()
print(sum(p.numel() for p in resnet_m.parameters()))

x = torch.randn(1, 3, 224, 224)
y = torch.randint(0, 1000, (1,)).long()

test_data = [(x, y)]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

resnet_m.eval()
y_hat = resnet_m(x)
loss = torch.nn.functional.cross_entropy(y_hat, y)
print(loss)
"""
# Instantiate the model
resnet_model = resnet18_moe()

# Print the number of parameters in the model
print(sum(p.numel() for p in resnet_model.parameters()))

# Create a random tensor with the same shape as the input to ResNet18
x = torch.randn(1, 3, 224, 224)
y = torch.randint(0, 1000, (1,)).long()

# Create a DataLoader for the test step
test_data = [(x, y)]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

# Run the test step
resnet_model.eval()
y_hat, gates, emb_y_hat = resnet_model(x)
loss = deepmoe_loss()(y_hat, emb_y_hat, y, gates)
print(loss)

# Testing for wide ResNet

# Instantiate the model
resnet_model = resnet18_moe(wide=True)

# Print the number of parameters in the model
print(sum(p.numel() for p in resnet_model.parameters()))

# Create a random tensor with the same shape as the input to ResNet18
x = torch.randn(1, 3, 224, 224)
y = torch.randint(0, 1000, (1,)).long()

# Create a DataLoader for the test step
test_data = [(x, y)]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

# Run the test step
resnet_model.eval()
y_hat, gates, emb_y_hat = resnet_model(x)
loss = deepmoe_loss()(y_hat, emb_y_hat, y, gates)
print(loss)