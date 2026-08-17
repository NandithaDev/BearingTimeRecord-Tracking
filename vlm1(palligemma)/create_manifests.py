import os
import json

# ==================================================
# PATHS
# ==================================================

dataset_root = "dataset_10000"
split_root = os.path.join(dataset_root, "split")
manifest_root = os.path.join(dataset_root, "manifests")

os.makedirs(manifest_root, exist_ok=True)


# ==================================================
# PROMPT
# ==================================================

PROMPT = (
    "Analyze this bearing-time record. "
    "Identify the number of targets and provide the trajectory "
    "of each target as a sequence of bearing values over time."
)


# ==================================================
# CREATE MANIFEST
# ==================================================

def create_manifest(split):

    images_folder = os.path.join(
        split_root,
        split,
        "images"
    )

    annotations_folder = os.path.join(
        split_root,
        split,
        "annotations"
    )

    output_file = os.path.join(
        manifest_root,
        f"{split}.jsonl"
    )

    image_files = sorted(
        f for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    count = 0

    with open(output_file, "w") as outfile:

        for image_file in image_files:

            # --------------------------------------------------
            # Corresponding JSON annotation
            # --------------------------------------------------

            simulation_id = os.path.splitext(image_file)[0].replace("btr_", "")
            json_file = f"scene_{simulation_id}.json"

            json_path = os.path.join(
                annotations_folder,
                json_file
            )

            if not os.path.exists(json_path):
                print(f"⚠️ Missing annotation: {json_file}")
                continue

            # --------------------------------------------------
            # Read annotation
            # --------------------------------------------------

            with open(json_path, "r") as f:
                scene = json.load(f)

            # --------------------------------------------------
            # Construct answer
            # --------------------------------------------------

            num_targets = scene["num_targets"]

            answer = f"There are {num_targets} targets."

            for i, target in enumerate(scene["targets"], start=1):

                trajectory = target["trajectory"]

                answer += (
                    f" Target {i} starts at bearing "
                    f"{target['start_bearing']} degrees and ends at "
                    f"{target['end_bearing']} degrees. "
                    f"Its trajectory is {trajectory}."
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
                "prompt": PROMPT,
                "answer": answer
            }

            outfile.write(
                json.dumps(entry) + "\n"
            )

            count += 1

    print(f"✅ {split}: {count} samples")
    print(f"   Saved to: {output_file}")


# ==================================================
# CREATE ALL SPLITS
# ==================================================

create_manifest("train")
create_manifest("val")
create_manifest("test")

print("\n🎯 Manifest generation completed!")