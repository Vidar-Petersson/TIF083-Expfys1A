import pandas as pd
import numpy as np
from sympy import Point, Circle, atan2, pi
import os
import matplotlib.pyplot as plt

# Constants for the experiment
hz = 500  # Camera's update frequency in Hz
mass_a = 0.0287262  # Mass of object A in kg
mass_b = 0.0281852  # Mass of object B in kg
radius_a = 0.0025  # Radius of object A in meters
radius_b = 0.0025  # Radius of object B in meters

# Function to calculate circle center and angle over time from three points
def circle_center(p1, p2, p3):
    center_x = []
    center_y = []
    angle = []
    for e1, e2, e3 in zip(p1, p2, p3):

        d1, d2, d3 = Point(e1[0], e1[1]), Point(e2[0], e2[1]), Point(e3[0], e3[1])  # Extract x, y positions from each point

        # Determine the center of the circle
        p_center = Circle(d1, d2, d3).center

        center_x.append(p_center.x.evalf())
        center_y.append(p_center.y.evalf())

        angle_p = atan2(d1.y - p_center.y, d1.x - p_center.x)  # Calculate angle between points and circle center

        # Normalize the angle to [0, 2*pi)
        if angle_p < 0:
            angle_p += 2 * pi

        angle.append(angle_p.evalf())
    return (center_x, center_y, angle)

dir = "DATA//raw//del2/"

for entry in os.listdir(dir):
    if entry.endswith(".tsv"):
        f_name = os.path.join(dir, entry)
        print(f_name)

        # Import TSV data from QTM
        df = pd.read_csv(f_name, delimiter="\t", skiprows=11)
        df.replace(0.0, np.nan, inplace=True)
        df.drop("Unnamed: 18", axis=1, inplace=True)

        # Extract and convert points from mm to meters
        p1 = np.column_stack((df["New 0000 X"], df["New 0000 Y"])) / 1000
        p2 = np.column_stack((df["New 0001 X"], df["New 0001 Y"])) / 1000
        p3 = np.column_stack((df["New 0002 X"], df["New 0002 Y"])) / 1000
        p4 = np.column_stack((df["New 0003 X"], df["New 0003 Y"])) / 1000
        p5 = np.column_stack((df["New 0004 X"], df["New 0004 Y"])) / 1000
        p6 = np.column_stack((df["New 0005 X"], df["New 0005 Y"])) / 1000

        points = np.array([p2, p3, p4, p5, p6])

        # Calculate distances using NumPy
        distance = np.sqrt(np.sum((points[:, 0] - p1[0]) ** 2, axis=1))

        # Find the indices of the two closest coordinates
        two_closest_indices = np.argsort(distance)[:2]
        not_a = np.setdiff1d(np.array([0, 1, 2, 3, 4]), two_closest_indices)

        # The two closest points belong to one circle, the rest belong to the other circle
        a1 = p1
        a2 = points[two_closest_indices[0]]
        a3 = points[two_closest_indices[1]]
        b1 = points[not_a[0]]
        b2 = points[not_a[1]]
        b3 = points[not_a[2]]

        # Calculate the centers and angles for both circles
        center_a_x, center_a_y, angle_a = circle_center(a1, a2, a3)  # For puck A
        print("Puck A done")
        center_b_x, center_b_y, angle_b = circle_center(b1, b2, b3)  # For puck B
        print("Puck B done")

        # Save the calculated data to a CSV file
        data = {
            "c_a_x": center_a_x,
            "c_a_y": center_a_y,
            "ang_a": angle_a,
            "c_b_x": center_b_x,
            "c_b_y": center_b_y,
            "ang_b": angle_b,
        }

        out_name = dir + "/centers//" + entry + ".csv"
        pd.DataFrame(data).to_csv(out_name, index=False)
