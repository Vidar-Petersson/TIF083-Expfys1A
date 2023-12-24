import pandas as pd
import numpy as np
import os

# Constants for the experiment
hz = 500  # Cameras' capture rate in Hz
interval = 0.2  # Time interval in seconds
mass_a = 0.2886  # Mass of object A in kg
mass_b = 0.3022  # Mass of object B in kg

# Define a class for a rider
class Rider:
    def __init__(self, x, y, mass):
        # Convert coordinates from millimeters to meters since the camera measures in mm
        x, y = x / 1000, y / 1000
        self.pos = np.column_stack((x, y))  # Position
        self.v = np.column_stack((np.gradient(x) * hz, np.gradient(y) * hz))  # Velocity vector
        self.p = np.column_stack(
            (np.gradient(x) * hz * mass, np.gradient(y) * hz * mass)
        )  # Momentum vector

# Directory containing data files
dir = "DATA//raw//del1//metall"

e_list = []
v_rel_list = []

# Loop through all measurement series and calculate relative velocity and elasticity
for entry in os.listdir(dir):
    if entry.endswith(".tsv"):
        f_name = os.path.join(dir, entry)

        # Read the data from the TSV file, skipping 11 rows
        df = pd.read_csv(f_name, delimiter="\t", skiprows=11)

        # Create Rider instances for object A and B
        rider_a = Rider(df["a X"], df["a Y"], mass_a)
        rider_b = Rider(df["b X"], df["b Y"], mass_b)

        # Calculate relative velocity
        v_rel = abs(rider_a.v - rider_b.v)
        v_rel = np.linalg.norm(v_rel, axis=1)  # Calculate the magnitude of the relative velocity vector

        # Find the frame index when collision occurs
        collision_frame = np.argmin(np.linalg.norm(rider_a.pos - rider_b.pos, axis=1))
        print(collision_frame)

        # Calculate elasticity
        e = abs(
            np.mean(v_rel[collision_frame + 1 : collision_frame + int(interval * hz)])
            / np.mean(v_rel[collision_frame - int(interval * hz) : collision_frame - 1])
        )

        # Check conditions for recording data
        e_list.append(e)
        v_rel_list.append(np.mean(v_rel[collision_frame - int(0.2 * hz) : collision_frame]))

# Save the collected data to a new CSV file
data = {
    "v_rel": v_rel_list,
    "e": e_list
}

pd.DataFrame(data).to_csv("DATA/del1_final_gummi.csv", index=False)
