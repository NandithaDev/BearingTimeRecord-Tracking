import torch
from model import UNet

model = UNet()

x = torch.randn(1, 3, 512, 512)

y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)