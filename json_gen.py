import os
import json 

json_folder = "scene_json"
os.makedirs(json_folder, exist_ok=True)

def get_visibility(snr):

    if snr >= 15:
        return "very strong"

    elif snr >= 8:
        return "strong"

    elif snr >= 0:
        return "moderate"

    else:
        return "weak"

def get_direction(velocity):

    if velocity > 0:
        return "increasing"

    elif velocity < 0:
        return "decreasing"

    else:
        return "stationary"
def get_motion(velocity):

    if velocity == 0:
        return "stationary"

    return "constant"

#main fn
def build_scene_json(simulation_id,
                     targets,
                     time_steps,
                     bearing_range=(0,180)):
        #creating scene
        scene = {
        "scene":{

            "simulation_id": simulation_id,
                                                    #we get all these parameters from the ground truth, here from the synthetic data
            "num_targets": len(targets),

            "time_steps": time_steps,

            "bearing_range": list(bearing_range)

        },

        "targets":[]
    }
        
        for idx, target in enumerate(targets, start=1):    #loooping thru to simulataneously get targets and index
                                                        #LEARN MORE ABT THIS PART !
            trajectory = target["trajectory"]

            velocity = int(target["velocity"])

            snr = int(target["snr"])

            target_json = {

                "target_id": idx,

                "trajectory": trajectory.tolist(),

                "start_bearing": int(trajectory[0]),

                "end_bearing": int(trajectory[-1]),

                "direction": get_direction(velocity),

                "motion": get_motion(velocity),

                "velocity": velocity,

                "duration": len(trajectory),

                "snr_db": snr,

                "visibility": get_visibility(snr)
        }

            scene["targets"].append(target_json)
            
        return scene
    
#save
def save_scene_json(scene):

    simulation_id = scene["scene"]["simulation_id"]

    filename = os.path.join(
        json_folder,
        f"scene_{simulation_id}.json"
    )

    with open(filename, "w") as f:

        json.dump(scene,
                  f,
                  indent=4)

    print(f"Saved {filename}")

