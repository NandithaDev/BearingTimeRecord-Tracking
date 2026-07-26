import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from model import UNet

# -----------------------------
# Device
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = UNet().to(device)
model.load_state_dict(torch.load("bearing_unet.pth", map_location=device))
model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# -----------------------------
# Load Noisy Image
# -----------------------------
image_path = "/Users/afwansha/Desktop/projects/DRDO_NPOL/bearingunet/dataset/noisy/MT_4Targets_486.jpg"   # Change this to any image

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)

output = output.squeeze().cpu().numpy()

# Convert CHW -> HWC
output = np.transpose(output, (1, 2, 0))

# Clamp values between 0 and 1
output = np.clip(output, 0, 1)

# Convert to uint8
output = (output * 255).astype(np.uint8)

# Save image
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite("prediction.jpg", output)

print("Prediction saved as prediction.jpg")