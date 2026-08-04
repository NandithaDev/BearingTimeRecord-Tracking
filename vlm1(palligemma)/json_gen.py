import os
import json


# ==================================================
# BUILD JSON
# ==================================================

def build_scene_json(
    
    targets,
    
):

    scene = {
        "num_targets": len(targets),
        "targets": []
    }

    for target in targets:

        trajectory = target["trajectory"]

        target_json = {

            "start_bearing": int(trajectory[0]),

            "end_bearing": int(trajectory[-1]),

            # Complete trajectory (50 values)
            "trajectory": trajectory.tolist()

        }

        scene["targets"].append(target_json)

    return scene


# ==================================================
# SAVE JSON
# ==================================================

def save_scene_json(scene, simulation_id, json_folder):

    os.makedirs(json_folder, exist_ok=True)

    filename = os.path.join(
        json_folder,
        f"scene_{simulation_id:05d}.json"
    )

    with open(filename, "w") as f:
        json.dump(scene, f, indent=4)

    print(f"Saved {filename}")