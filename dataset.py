import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BearingDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        # Get all noisy image filenames
        self.images = sorted(os.listdir(noisy_dir))

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Get noisy filename
        filename = self.images[idx]

        # Paths
        noisy_path = os.path.join(self.noisy_dir, filename)

        # Ground truth filename has G_ prefix
        gt_filename = "G_" + filename
        gt_path = os.path.join(self.gt_dir, gt_filename)

        # Read images
        noisy = cv2.imread(noisy_path)
        gt = cv2.imread(gt_path)

        # Error checking
        if noisy is None:
            raise FileNotFoundError(f"Could not load noisy image: {noisy_path}")

        if gt is None:
            raise FileNotFoundError(f"Could not load ground truth image: {gt_path}")

        # Convert BGR → RGB
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        # Resize and convert to tensor
        noisy = self.transform(noisy)
        gt = self.transform(gt)

        return noisy, gt