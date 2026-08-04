import os
import json

ROOT = "dataset"      # Change if your dataset folder has another name

PROMPT = "Predict the target trajectories."


def create_manifest(split):

    image_dir = os.path.join(ROOT, split, "images")
    annotation_dir = os.path.join(ROOT, split, "annotations")

    output_file = os.path.join(ROOT, f"{split}.jsonl")

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    )

    with open(output_file, "w") as out:

        for image_name in image_files:

            sim_id = image_name.replace("btr_", "").replace(".jpg", "")

            annotation_name = f"scene_{sim_id}.json"

            annotation_path = os.path.join(annotation_dir, annotation_name)

            with open(annotation_path, "r") as f:
                annotation = json.load(f)

            sample = {
                "image": os.path.join(split, "images", image_name),
                "prefix": PROMPT,
                "suffix": json.dumps(annotation, separators=(",", ":"))
            }

            out.write(json.dumps(sample) + "\n")

    print(f"Created {output_file}")


for split in ["train", "val", "test"]:
    create_manifest(split)