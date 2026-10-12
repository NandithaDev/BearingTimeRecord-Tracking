import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap

from json_gen import build_scene_json, save_scene_json

# ==================================================
# CUSTOM COLORMAP
# ==================================================

black_to_green = LinearSegmentedColormap.from_list(
    "black_to_green",
    ["black", "green"]
)

# ==================================================
# OUTPUT FOLDERS
# ==================================================

output_folder = "dataset_10000"

os.makedirs(output_folder, exist_ok=True)

btr_img_folder = os.path.join(output_folder, "btr_images")
json_folder = os.path.join(output_folder, "scene_json")

for folder in [
    
    btr_img_folder,
    
    json_folder,
]:
    
    os.makedirs(folder, exist_ok=True)

# ==================================================
# SIMULATION PARAMETERS
# ==================================================

final_sim_no = 10000      # Generate 500 simulations

T = 50
c = 1500
omega = 1000
M = 32

d = 0.45 * c / omega

Theta = np.arange(0, 181)

# ==================================================
# AUTO RESUME
# ==================================================



sim_numbers = set()

for sim_num in range(1, final_sim_no + 1):

    image_ok = os.path.exists(
    os.path.join(
        btr_img_folder,
        f"btr_{sim_num:05d}.jpg"
    )
)

    json_ok = os.path.exists(
        os.path.join(
            json_folder,
            f"scene_{sim_num:05d}.json"
     )
    )

    if image_ok and json_ok:
        sim_numbers.add(sim_num)



start_sim_no = 1

while start_sim_no in sim_numbers:
    start_sim_no += 1

if len(sim_numbers) > 0:
    print(f"🔁 Resuming from simulation {start_sim_no}")
else:
    print("🆕 Starting fresh simulation")



# ==================================================
# MAIN SIMULATION LOOP
# ==================================================

def sim_exists(sim_num):

    image_ok = os.path.exists(
        os.path.join(
            btr_img_folder,
            f"btr_{sim_num:05d}.jpg"
        )
    )

    json_ok = os.path.exists(
        os.path.join(
            json_folder,
            f"scene_{sim_num:05d}.json"
        )
    )

    return image_ok and json_ok

for simulation_num in range(start_sim_no, final_sim_no + 1):

    # Skip simulations that are already complete
    if sim_exists(simulation_num):
        print(f"⏩ Skipping Simulation {simulation_num}")
        continue

    # Initialize Bearing-Time Record matrix
    Z = np.zeros((T, len(Theta)))

    # ==================================================
    # RANDOMLY GENERATE TARGETS
    # ==================================================

    num_targets = np.random.randint(0, 5)      # 0–4 targets

    targets = []

    for _ in range(num_targets):

        valid = False

        while not valid:

            # Random starting DOA
            start_doa = np.random.randint(0, 181)

            # Random angular velocity
            velocity = np.random.randint(-2, 3)

            # Target trajectory over time
            trajectory = start_doa + velocity * np.arange(T)

            # Ensure target never leaves the field of view
            if np.all((trajectory >= 0) & (trajectory <= 180)):
                valid = True
                trajectory = trajectory.astype(int)

        # Random Signal-to-Noise Ratio
        snr = np.random.randint(-10, 20)

        targets.append({
            "trajectory": trajectory,
            "snr": snr,
            "velocity": velocity
        })

    

    # ==================================================
    # GENERATE BTR
    # ==================================================

    for t in range(T):

        y_total = np.zeros(M, dtype=complex)

        for target in targets:

            doa = target["trajectory"][t]
            snr = target["snr"]

            y = np.exp(
                -1j * 2 * np.pi * omega * d / c
                * np.cos(np.radians(doa))
                * np.arange(M)
            )

            y_total += 10 ** (snr / 20) * y + np.random.randn(M)

        # Steering matrix
        S = np.exp(
            1j * 2 * np.pi * omega * d / c
            * np.cos(np.radians(Theta))[:, None]
            * np.arange(M)
        )

        power = np.abs(np.dot(y_total, S.T)) ** 2

        # Avoid log(0)
        power = np.maximum(power, 1e-12)

        Z[t, :] = 10 * np.log10(power)
        
        
    # ==================================================
    # PLOT AND SAVE BTR
    # ==================================================

    X = np.arange(1, T + 1)
    Theta_grid, X_grid = np.meshgrid(Theta, X)

    plt.figure()
    plt.pcolor(
        Theta_grid,
        X_grid,
        Z,
        shading="auto",
        cmap=black_to_green
    )

    plt.xlabel("Bearing")
    plt.ylabel("Time")


    def save_pair(sim_num):

        sim_id = f"{sim_num:05d}"

        filename = os.path.join(
            btr_img_folder,
            f"btr_{sim_id}.jpg"
        )

        plt.title(f"btr_{sim_id}")

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

    save_pair(simulation_num)
    plt.close()


    # # ==================================================
    # # GENERATE GROUND TRUTH
    # # ==================================================

    # Z_binary = np.zeros_like(Z)

    # for t in range(T):

    #     for target_id, target in enumerate(targets):

    #         doa = target["trajectory"][t]

    #         idx = np.argmin(np.abs(Theta - doa))

    #         # Label each target with a different integer
    #         Z_binary[t, idx] = target_id + 1


    # plt.figure()

    # plt.pcolor(
    #     Theta_grid,
    #     X_grid,
    #     Z_binary,
    #     shading="auto",
    #     cmap=black_to_green
    # )

    # plt.xlabel("Bearing")
    # plt.ylabel("Time")

    # save_pair(simulation_num, Z_binary, prefix="G_")
    # plt.close()


    # ==================================================
    # SAVE JSON DESCRIPTION
    # ==================================================

    scene = build_scene_json(
        
        targets=targets,
        
    )

    save_scene_json(scene, simulation_num, json_folder)


    print(f"✅ Saved Simulation {simulation_num}: {num_targets} targets")



print(
    f"🎯 Dataset generation completed successfully! "
    f"(Started from simulation {start_sim_no})"
)