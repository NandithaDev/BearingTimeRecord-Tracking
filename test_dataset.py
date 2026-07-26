from dataset import BearingDataset

dataset = BearingDataset(
    "noisy",
    "ground_truth"
)

print("Number of images:", len(dataset))

noisy, gt = dataset[0]

print("Noisy shape:", noisy.shape)
print("Ground truth shape:", gt.shape)