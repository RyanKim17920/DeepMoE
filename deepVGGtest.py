import torch
import pytorch_lightning as pl
from deepVGG import VGG16, VGGtoDeepMoe
from deepmoe_utils import deepmoe_loss

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam 

# Define a LightningModule for your model
class DeepMoEModel(pl.LightningModule):
    def __init__(self):
        super(DeepMoEModel, self).__init__()
        self.model = VGGtoDeepMoe(VGG16())

    def forward(self, x, predict=False):
        y_hat, emb_y_hat, gates = self.model(x, predict=predict)
        return y_hat, emb_y_hat, gates

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_hat, emb_y_hat, gates = self.forward(x, predict=False)
        loss = deepmoe_loss(y_hat, emb_y_hat, y, gates)
        self.log('test_loss', loss)
        return loss


# Instantiate the model
deepmoe_model = DeepMoEModel()

# Print the number of parameters in the model
print(sum(p.numel() for p in deepmoe_model.parameters()))

# Create a random tensor with the same shape as the input to VGG16
x = torch.randn(1, 3, 224, 224)
y = torch.randint(0, 1000, (1,)).long()

# Create a DataLoader for the test step
test_data = [(x, y)]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

# Configure the trainer with DeepSpeed
trainer = pl.Trainer(
    precision="16",
    strategy="deepspeed_stage_3_offload",
)

# Run the test step
trainer.test(deepmoe_model, test_loader)
