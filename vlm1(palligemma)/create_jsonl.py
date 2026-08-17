import os
import json


# ==================================================
# DATASET PATH
# ==================================================

dataset_folder = "dataset_10000/split"


# ==================================================
# PROMPT
# ==================================================

PROMPT = (
    "Analyze the bearing-time record and identify all targets. "
    "For each target, provide its start bearing, end bearing, "
    "and complete bearing trajectory over time."
)


# ==================================================
# CREATE JSONL
# ==================================================

def create_jsonl(split):

    split_folder = os.path.join(dataset_folder, split)

    image_folder = os.path.join(
        split_folder,
        "images"
    )

    annotation_folder = os.path.join(
        split_folder,
        "annotations"
    )

    output_file = os.path.join(
        dataset_folder,
        f"{split}.jsonl"
    )

    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.endswith(".jpg")
    )

    count = 0

    with open(output_file, "w") as outfile:

        for image_file in image_files:

            # --------------------------------------------------
            # Get simulation ID
            # btr_00001.jpg -> 00001
            # --------------------------------------------------

            sim_id = image_file.replace(
                "btr_", ""
            ).replace(
                ".jpg", ""
            )

            annotation_file = f"scene_{sim_id}.json"

            annotation_path = os.path.join(
                annotation_folder,
                annotation_file
            )

            if not os.path.exists(annotation_path):
                print(
                    f"⚠️ Missing annotation: {annotation_file}"
                )
                continue

            # --------------------------------------------------
            # Read ground-truth JSON
            # --------------------------------------------------

            with open(annotation_path, "r") as f:
                annotation = json.load(f)

            # --------------------------------------------------
            # Convert answer to a compact JSON string
            # --------------------------------------------------

            answer = json.dumps(
                annotation,
                separators=(",", ":")
            )

            # --------------------------------------------------
            # JSONL entry
            # --------------------------------------------------

            entry = {
                "image": os.path.join(
                    split,
                    "images",
                    image_file
                ),
                "prefix": PROMPT,
                "suffix": answer
            }

            outfile.write(
                json.dumps(entry) + "\n"
            )

            count += 1

    print(
        f"✅ {split}: {count} samples → {output_file}"
    )


# ==================================================
# CREATE ALL SPLITS
# ==================================================

create_jsonl("train")
create_jsonl("val")
create_jsonl("test")

print("\n🎯 JSONL generation completed!")