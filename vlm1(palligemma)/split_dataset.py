import os
import random
import shutil

# ==================================================
# PATHS
# ==================================================

dataset_folder = "dataset_10000"

image_folder = os.path.join(dataset_folder, "btr_images")
json_folder = os.path.join(dataset_folder, "scene_json")

split_folder = os.path.join(dataset_folder, "split")

# ==================================================
# SPLIT RATIOS
# ==================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

# ==================================================
# GET SIMULATION IDs
# ==================================================

image_files = [
    f for f in os.listdir(image_folder)
    if f.endswith(".jpg")
]

simulation_ids = []

for image_file in image_files:

    # btr_00001.jpg -> 00001
    sim_id = image_file.replace("btr_", "").replace(".jpg", "")

    json_file = f"scene_{sim_id}.json"

    # Only include samples where BOTH exist
    if os.path.exists(os.path.join(json_folder, json_file)):
        simulation_ids.append(sim_id)

print(f"Total paired samples: {len(simulation_ids)}")

# ==================================================
# RANDOM SHUFFLE
# ==================================================

random.shuffle(simulation_ids)

total = len(simulation_ids)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_ids = simulation_ids[:train_end]
val_ids = simulation_ids[train_end:val_end]
test_ids = simulation_ids[val_end:]

print(f"Train: {len(train_ids)}")
print(f"Validation: {len(val_ids)}")
print(f"Test: {len(test_ids)}")

# ==================================================
# CREATE DIRECTORIES
# ==================================================

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(split_folder, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(split_folder, split, "annotations"),
        exist_ok=True
    )

# ==================================================
# COPY FILES
# ==================================================

def copy_split(simulation_ids, split):

    for sim_id in simulation_ids:

        image_source = os.path.join(
            image_folder,
            f"btr_{sim_id}.jpg"
        )

        json_source = os.path.join(
            json_folder,
            f"scene_{sim_id}.json"
        )

        image_destination = os.path.join(
            split_folder,
            split,
            "images",
            f"btr_{sim_id}.jpg"
        )

        json_destination = os.path.join(
            split_folder,
            split,
            "annotations",
            f"scene_{sim_id}.json"
        )

        shutil.copy2(image_source, image_destination)
        shutil.copy2(json_source, json_destination)


# ==================================================
# RUN SPLIT
# ==================================================

copy_split(train_ids, "train")
copy_split(val_ids, "val")
copy_split(test_ids, "test")

print("\n✅ Dataset split completed!")