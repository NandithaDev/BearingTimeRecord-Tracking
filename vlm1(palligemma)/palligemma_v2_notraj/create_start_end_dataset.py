import os
import json

SOURCE_ROOT = "dataset_10000"
TARGET_ROOT = "dataset_start_end"

PROMPT = (
    "Analyze this bearing-time record. "
    "Identify the number of targets and provide the start and end "
    "bearing of each target."
)


def create_split(split):
    source_images = os.path.join(SOURCE_ROOT, "split", split, "images")
    source_annotations = os.path.join(SOURCE_ROOT, "split", split, "annotations")

    target_images = os.path.join(TARGET_ROOT, "split", split, "images")
    target_annotations = os.path.join(TARGET_ROOT, "split", split, "annotations")

    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_annotations, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(source_images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    count = 0

    for image_file in image_files:

        # btr_07151.jpg -> 07151 -> scene_07151.json
        simulation_id = os.path.splitext(image_file)[0].replace("btr_", "")
        annotation_file = f"scene_{simulation_id}.json"

        source_json = os.path.join(
            source_annotations,
            annotation_file
        )

        if not os.path.exists(source_json):
            print(f"⚠️ Missing annotation: {annotation_file}")
            continue

        # --------------------------------------------------
        # Reuse image using hard link
        # --------------------------------------------------

        target_image = os.path.join(target_images, image_file)

        if not os.path.exists(target_image):
            os.link(
                os.path.join(source_images, image_file),
                target_image
            )

        # --------------------------------------------------
        # Create simplified annotation
        # --------------------------------------------------

        with open(source_json, "r") as f:
            scene = json.load(f)

        new_scene = {
            "num_targets": scene["num_targets"],
            "targets": []
        }

        for target in scene["targets"]:
            new_scene["targets"].append({
                "start_bearing": int(target["start_bearing"]),
                "end_bearing": int(target["end_bearing"])
            })

        target_json = os.path.join(
            target_annotations,
            annotation_file
        )

        with open(target_json, "w") as f:
            json.dump(new_scene, f, indent=4)

        count += 1

    print(f"✅ {split}: {count} samples")


def create_manifest(split):
    images_folder = os.path.join(
        TARGET_ROOT, "split", split, "images"
    )

    annotations_folder = os.path.join(
        TARGET_ROOT, "split", split, "annotations"
    )

    manifest_folder = os.path.join(
        TARGET_ROOT, "manifests"
    )

    os.makedirs(manifest_folder, exist_ok=True)

    output_file = os.path.join(
        manifest_folder,
        f"{split}.jsonl"
    )

    image_files = sorted(
        f for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    count = 0

    with open(output_file, "w") as outfile:

        for image_file in image_files:

            simulation_id = os.path.splitext(image_file)[0].replace("btr_", "")
            annotation_file = f"scene_{simulation_id}.json"

            json_path = os.path.join(
                annotations_folder,
                annotation_file
            )

            if not os.path.exists(json_path):
                print(f"⚠️ Missing annotation: {annotation_file}")
                continue

            with open(json_path, "r") as f:
                scene = json.load(f)

            num_targets = scene["num_targets"]

            answer = f"There are {num_targets} targets."

            for i, target in enumerate(scene["targets"], start=1):
                answer += (
                    f" Target {i} starts at bearing "
                    f"{target['start_bearing']} degrees and ends at "
                    f"{target['end_bearing']} degrees."
                )

            entry = {
                "image": os.path.join(
                    split,
                    "images",
                    image_file
                ),
                "prompt": PROMPT,
                "answer": answer
            }

            outfile.write(json.dumps(entry) + "\n")
            count += 1

    print(f"✅ Manifest {split}: {count} samples")


# ==================================================
# CREATE DATASET
# ==================================================

for split in ["train", "val", "test"]:
    create_split(split)

for split in ["train", "val", "test"]:
    create_manifest(split)

print("\n🎯 Start/end dataset creation completed!")
