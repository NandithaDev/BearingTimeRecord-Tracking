import os
import json
import ollama

#configure
JSON_FOLDER = "scene_json"
CAPTION_FOLDER = "captions"
MODEL_NAME = "gemma3:4b"

os.makedirs(CAPTION_FOLDER, exist_ok=True)


# BUILD PROMPT

def build_prompt(scene):

    return f"""
You are an underwater passive sonar analyst.

You are given structured information about detected targets.

Write a natural language report describing the sonar scene.

Requirements:
- Mention the total number of targets.
- Describe each target separately.
- Mention whether it is stationary or moving.
- Mention its starting and ending bearing.
- Mention whether the signal is weak, moderate, strong or very strong.
- Do NOT invent any information.      
- Use professional but readable English.
- Keep the report within one short paragraph.

Scene JSON:

{json.dumps(scene, indent=2)}

Report:
"""


# GENERATE CAPTION USING GEMMA


def generate_caption(prompt):

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# MAIN

print("Working Directory :", os.getcwd())
print("Looking for JSONs :", os.path.abspath(JSON_FOLDER))

if not os.path.exists(JSON_FOLDER):
    raise FileNotFoundError(
        f"JSON folder not found: {JSON_FOLDER}"
    )

files = sorted(
    [
        f for f in os.listdir(JSON_FOLDER)
        if f.endswith(".json")
    ]
)

print(f"Found {len(files)} JSON files.\n")

if len(files) == 0:
    print("No JSON files found.")
    exit()


# LOOP THROUGH ALL JSON FILES

for index, file in enumerate(files, start=1):

    try:

        json_path = os.path.join(JSON_FOLDER, file)

        with open(json_path, "r") as f:
            scene = json.load(f)

        sim_id = scene["scene"]["simulation_id"]

        caption_file = os.path.join(
            CAPTION_FOLDER,
            f"caption_{sim_id}.txt"
        )

        # Skip already generated captions
        if os.path.exists(caption_file):
            print(f"[{index}/{len(files)}]  Skipping Scene {sim_id}")
            continue

        prompt = build_prompt(scene)

        caption = generate_caption(prompt)

        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)

        print(f"[{index}/{len(files)}] Scene {sim_id}")

    except Exception as e:

        print(f"\n Error processing {file}")
        print(e)
        print()

print("\n Caption generation complete!")