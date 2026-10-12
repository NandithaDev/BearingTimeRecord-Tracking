
import os
import json
import random

import torch
from PIL import Image
# import matplotlib.pyplot as plt

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/workspace/vlm1(palligemma)"

BASE_MODEL = "google/paligemma-3b-pt-224"

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "paligemma_json_final"
)

DATASET_ROOT = os.path.join(
    PROJECT_DIR,
    "dataset_start_end"
)

# Use TEST for final evaluation.
# Change to "val" only if you specifically want validation samples.
SPLIT = "test"

NUM_SAMPLES = 10

RANDOM_SEED = 42

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "split",
    SPLIT,
    "images"
)

ANNOTATION_DIR = os.path.join(
    DATASET_ROOT,
    "split",
    SPLIT,
    "annotations"
)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("PaliGemma JSON Model Evaluation")
print("=" * 70)

print(f"Model       : {MODEL_DIR}")
print(f"Dataset     : {DATASET_ROOT}")
print(f"Split       : {SPLIT}")
print(f"Samples     : {NUM_SAMPLES}")
print(f"Random seed : {RANDOM_SEED}")
print(f"Device      : {device}")

if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")

print("=" * 70)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = PaliGemmaProcessor.from_pretrained(
    MODEL_DIR
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("Loading base PaliGemma model...")

base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

base_model = base_model.to(device)


# ============================================================
# LOAD LoRA ADAPTER
# ============================================================

print("Loading trained LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR
)

model.eval()

print("✅ Model loaded.")


# ============================================================
# FIND TEST IMAGES
# ============================================================

image_files = sorted(
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
)

print(f"\nTotal images in {SPLIT}: {len(image_files)}")


# ============================================================
# RANDOM SAMPLING
# ============================================================

random.seed(RANDOM_SEED)

num_to_sample = min(
    NUM_SAMPLES,
    len(image_files)
)

sampled_images = random.sample(
    image_files,
    num_to_sample
)

print(
    f"Randomly selected {num_to_sample} images."
)


# ============================================================
# METRICS
# ============================================================

valid_json_count = 0
target_count_correct = 0


# ============================================================
# EVALUATION LOOP
# ============================================================

for sample_idx, image_file in enumerate(
    sampled_images,
    start=1
):

    print("\n")
    print("=" * 70)
    print(
        f"SAMPLE {sample_idx}/{num_to_sample}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    simulation_id = (
        os.path.splitext(image_file)[0]
        .replace("btr_", "")
    )

    annotation_file = (
        f"scene_{simulation_id}.json"
    )

    annotation_path = os.path.join(
        ANNOTATION_DIR,
        annotation_file
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    with open(
        annotation_path,
        "r"
    ) as f:

        ground_truth = json.load(f)

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = (
        "<image>"
        "Analyze this bearing-time record. "
        "Identify the number of targets and provide the "
        "start and end bearing of each target. "
        "Return ONLY valid JSON in this format: "
        '{"num_targets": 0, '
        '"targets": [{"start_bearing": 0, '
        '"end_bearing": 0}]}'
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    # Move tensors to GPU
    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    # --------------------------------------------------------
    # Remove prompt tokens
    # --------------------------------------------------------

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[-1]:
    ]

    prediction_text = processor.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------------
    # Parse prediction as JSON
    # --------------------------------------------------------

    try:

        prediction = json.loads(
            prediction_text
        )

        valid_json = True

        valid_json_count += 1

    except json.JSONDecodeError:

        prediction = None

        valid_json = False

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"\nImage: {image_file}")

    print("\nGROUND TRUTH:")
    print(
        json.dumps(
            ground_truth,
            indent=2
        )
    )

    print("\nRAW MODEL OUTPUT:")
    print(prediction_text)

    if valid_json:

        print("\nPARSED PREDICTION:")
        print(
            json.dumps(
                prediction,
                indent=2
            )
        )

        # Target count check
        if (
            prediction.get("num_targets")
            == ground_truth.get("num_targets")
        ):

            target_count_correct += 1

            print(
                "\n✅ Target count: CORRECT"
            )

        else:

            print(
                "\n❌ Target count: WRONG"
            )

    else:

        print(
            "\n❌ INVALID JSON"
        )

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    # plt.figure(
    #     figsize=(12, 5)
    # )

    # plt.imshow(image)

    # plt.title(
    #     f"{image_file} — "
    #     f"Ground Truth: "
    #     f"{ground_truth['num_targets']} targets"
    # )

    # plt.axis("off")

    # plt.show()


# ============================================================
# FINAL METRICS
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Samples evaluated : {num_to_sample}"
)

print(
    f"Valid JSON        : "
    f"{valid_json_count}/{num_to_sample}"
)

print(
    f"JSON validity     : "
    f"{valid_json_count / num_to_sample * 100:.2f}%"
)

print(
    f"Correct target count: "
    f"{target_count_correct}/{num_to_sample}"
)

print(
    f"Target count accuracy: "
    f"{target_count_correct / num_to_sample * 100:.2f}%"
)

print("=" * 70)
