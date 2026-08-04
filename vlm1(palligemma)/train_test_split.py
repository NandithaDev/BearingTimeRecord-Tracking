
import os
import shutil
import random

# ===============================
# SETTINGS
# ===============================

random.seed(42)

image_dir = "test data/btr_images"
json_dir = "test data/scene_json"

output_root = "vlm_dataset"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ===============================
# CREATE FOLDERS
# ===============================

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, split, "labels"), exist_ok=True)

# ===============================
# FIND VALID IMAGE-JSON PAIRS
# ===============================

samples = []

for image_file in os.listdir(image_dir):

    if not image_file.endswith(".jpg"):
        continue

    sim_id = image_file.split("_")[1].split(".")[0]

    json_file = f"scene_{sim_id}.json"

    if os.path.exists(os.path.join(json_dir, json_file)):
        samples.append((image_file, json_file))

print(f"Found {len(samples)} valid image/json pairs.")

# ===============================
# SHUFFLE
# ===============================

random.shuffle(samples)

# ===============================
# SPLIT
# ===============================

n = len(samples)

train_end = int(train_ratio * n)
val_end = train_end + int(val_ratio * n)

train = samples[:train_end]
val = samples[train_end:val_end]
test = samples[val_end:]

print(f"Train : {len(train)}")
print(f"Val   : {len(val)}")
print(f"Test  : {len(test)}")

# ===============================
# COPY FILES
# ===============================

def copy_split(split_name, split_data):

    image_out = os.path.join(output_root, split_name, "images")
    label_out = os.path.join(output_root, split_name, "labels")

    for image_file, json_file in split_data:

        shutil.copy2(
            os.path.join(image_dir, image_file),
            os.path.join(image_out, image_file)
        )

        shutil.copy2(
            os.path.join(json_dir, json_file),
            os.path.join(label_out, json_file)
        )


copy_split("train", train)
copy_split("val", val)
copy_split("test", test)

print("Dataset split complete.")