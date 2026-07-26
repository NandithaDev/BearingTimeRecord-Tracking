import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import BearingDataset
from model import UNet

# ===============================
# Device (Mac GPU / CPU)
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ===============================
# Dataset
# ===============================
dataset = BearingDataset(
    "noisy",
    "ground_truth"
)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# ===============================
# Model
# ===============================
model = UNet().to(device)

# ===============================
# Loss Function
# ===============================
criterion = nn.MSELoss()

# ===============================
# Optimizer
# ===============================
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ===============================
# Training Loop
# ===============================
epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0

    progress = tqdm(train_loader)

    for noisy, gt in progress:

        noisy = noisy.to(device)
        gt = gt.to(device)

        prediction = model(noisy)

        loss = criterion(prediction, gt)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{epochs}"
        )

        progress.set_postfix(
            loss=loss.item()
        )

    print(
        f"Epoch {epoch+1} Loss = {running_loss/len(train_loader):.6f}"
    )

# ===============================
# Save Model
# ===============================
torch.save(
    model.state_dict(),
    "bearing_unet.pth"
)

print("Training Finished!")