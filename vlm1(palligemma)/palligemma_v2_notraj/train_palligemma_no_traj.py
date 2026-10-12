
import os
import json
from PIL import Image

import torch
from datasets import load_dataset

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = "/workspace/vlm1(palligemma)"

DATASET_ROOT = os.path.join(
    PROJECT_DIR,
    "dataset_start_end"
)

MANIFEST_ROOT = os.path.join(
    DATASET_ROOT,
    "manifests"
)

SPLIT_ROOT = os.path.join(
    DATASET_ROOT,
    "split"
)

TRAIN_FILE = os.path.join(
    MANIFEST_ROOT,
    "train.jsonl"
)

VAL_FILE = os.path.join(
    MANIFEST_ROOT,
    "val.jsonl"
)

TEST_FILE = os.path.join(
    MANIFEST_ROOT,
    "test.jsonl"
)


# Fresh output directories for the JSON experiment
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "paligemma_json_output"
)

FINAL_DIR = os.path.join(
    PROJECT_DIR,
    "paligemma_json_final"
)


MODEL_ID = "google/paligemma-3b-pt-224"


# ============================================================
# GPU INFO
# ============================================================

print("=" * 60)
print("PaliGemma Start/End JSON Training")
print("=" * 60)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"GPU count       : {torch.cuda.device_count()}")

print(f"Dataset         : {DATASET_ROOT}")
print(f"Train manifest  : {TRAIN_FILE}")
print(f"Val manifest    : {VAL_FILE}")
print(f"Test manifest   : {TEST_FILE}")
print(f"Output          : {OUTPUT_DIR}")
print(f"Final model     : {FINAL_DIR}")

print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
        "test": TEST_FILE,
    }
)

print(dataset)


# ============================================================
# VALIDATE JSON ANSWERS
# ============================================================

print("\nValidating structured JSON answers...")

for split in ["train", "validation", "test"]:

    count = 0

    for example in dataset[split]:

        # Parse the answer string as JSON
        answer = json.loads(example["answer"])

        # Required top-level fields
        assert "num_targets" in answer
        assert "targets" in answer

        # Number of targets must match target list
        assert answer["num_targets"] == len(answer["targets"])

        # Validate every target
        for target in answer["targets"]:

            assert "start_bearing" in target
            assert "end_bearing" in target

            # Make absolutely sure trajectory is not present
            assert "trajectory" not in target

        count += 1

    print(
        f"  {split}: {count} valid JSON answers"
    )

print("✅ All dataset answers are valid structured JSON.")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading PaliGemma processor...")

processor = PaliGemmaProcessor.from_pretrained(
    MODEL_ID
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading PaliGemma model...")

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False


# ============================================================
# DATA COLLATOR
# ============================================================

def collate_fn(examples):

    images = []
    prompts = []
    answers = []

    for example in examples:

        image_path = os.path.join(
            SPLIT_ROOT,
            example["image"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        images.append(image)

        # PaliGemma expects the image token in the prompt
        prompts.append(
            "<image>" + example["prompt"]
        )

        # Structured JSON answer
        answers.append(
            example["answer"]
        )

    inputs = processor(
        text=prompts,
        images=images,
        suffix=answers,
        padding="longest",
        return_tensors="pt",
    )

    # Ignore padding tokens when calculating loss
    if "labels" in inputs:

        inputs["labels"][
            inputs["labels"]
            == processor.tokenizer.pad_token_id
        ] = -100

    return inputs


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    optim="adamw_torch",

    weight_decay=0.01,

    warmup_ratio=0.03,

    lr_scheduler_type="cosine",

    bf16=True,

    eval_strategy="steps",

    eval_steps=500,

    save_strategy="steps",

    save_steps=500,

    save_total_limit=2,

    logging_steps=10,

    dataloader_num_workers=4,

    gradient_checkpointing=True,

    load_best_model_at_end=True,

    report_to="none",

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["validation"],

    data_collator=collate_fn,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

trainer.save_model(
    FINAL_DIR
)

processor.save_pretrained(
    FINAL_DIR
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Final model saved to:")
print(FINAL_DIR)

