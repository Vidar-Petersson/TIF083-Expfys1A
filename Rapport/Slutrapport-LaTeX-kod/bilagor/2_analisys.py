import pandas as pd
import numpy as np
from sympy import pi
import os

# Constants for the experiment
hz = 500  # Camera update frequency
mass_a = 0.0287262  # Mass of object A in kg
radius_a = 0.025  # Radius of object A in meters
mass_b = 0.0281852  # Mass of object B in kg
radius_b = 0.025  # Radius of object B in meters

# Create a class for discs and calculate all relevant parameters
class Disk:
    def __init__(self, x, y, mass, radius, angle_df):
        inertia = 0.5 * mass * radius ** 2  # Moment of inertia
        angle = np.array(angle_df)  # Create a new instance of the dataframe

        # Compensate for the jump in arctan
        for i in range(len(angle) - 1):
            if angle[i + 1] - angle[i] > 3:
                for j in range(i + 1, len(angle)):
                    angle[j] -= 2 * pi
            elif angle[i + 1] - angle[i] < -3:
                for j in range(i + 1, len(angle)):
                    angle[j] += 2 * pi

        v_x = np.gradient(x) * hz
        v_y = np.gradient(y) * hz
        omega = np.gradient(angle) * hz

        self.pos = np.column_stack((x, y))  # Position vector
        self.omega = omega  # Angular velocity
        self.v = np.column_stack((v_x, v_y))  # Velocity
        self.p = np.column_stack((v_x * mass, v_y * mass))  # Linear momentum
        self.l = inertia * omega  # Angular momentum
        self.t = (
            np.sqrt(v_x ** 2 + v_y ** 2) ** 2 * mass * 0.5 + 0.5 * inertia * omega ** 2
        )  # Kinetic energy
        self.a = np.column_stack(
            (np.gradient(v_x), np.gradient(v_y))
        )  # Acceleration

dir = "DATA//raw//del2//centers/"

interval = 0.2 #seconds

for entry in os.listdir(dir):
    if entry.endswith(".csv"):
        f_name = os.path.join(dir, entry)
        print(f_name)

        df = pd.read_csv(f_name)

        center_a_x = df["c_a_x"]
        center_a_y = df["c_a_y"]
        center_b_x = df["c_b_x"]
        center_b_y = df["c_b_y"]
        angle_a = df["ang_a"]
        angle_b = df["ang_b"]

        # Create objects
        disk_a = Disk(center_a_x, center_a_y, mass_a, radius_a, angle_a)  # Create object for puck A
        disk_b = Disk(center_b_x, center_b_y, mass_b, radius_b, angle_b)  # Create object for puck B

        collision_f = np.argmin(np.linalg.norm(disk_a.pos - disk_b.pos, axis=1))  # Frame index of collision

        # Ensure that disk B is the stationary disk, correct if names are swapped
        if (
            np.linalg.norm(disk_a.v, axis=1)[collision_f - 100]
            < np.linalg.norm(disk_b.v, axis=1)[collision_f - 100]
        ):
            disk_a, disk_b = disk_b, disk_a

        d_vector = (disk_b.pos - disk_a.pos)[collision_f]
        v_vector = disk_a.v[collision_f - 2]

        theta = np.arccos(
            np.dot(d_vector, v_vector)
            / (np.linalg.norm(d_vector) * np.linalg.norm(v_vector))
        )  # Angle between velocity and difference vector
        dist = (
            np.sin(theta) * np.linalg.norm(d_vector) / (radius_a + radius_b)
        )  # Calculate d, which is [0,1] the collision point to be varied

        