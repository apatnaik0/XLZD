import pandas as pd 
import numpy as np
from numpy.random import uniform
from shell_probs import find_shells

def create_emulated_data(height, radius, width, num_points):
    # Generate data around the sides
    rand_r = np.sqrt(uniform(radius**2, (radius+width)**2, num_points))
    rand_theta = uniform(0, 2*np.pi, num_points)
    rand_z = uniform(0, height, num_points)
    # Convert to x/y
    rand_x = rand_r*np.cos(rand_theta)
    rand_y = rand_r*np.sin(rand_theta)
    side_points = np.column_stack([rand_x, rand_y, rand_z])

    # Generate data for the bottom "cap" of the cylinder
    rand_r = np.sqrt(uniform(0, (radius+width)**2, num_points))
    rand_theta = uniform(0, 2*np.pi, num_points)
    rand_z = uniform(-width, 0, num_points)
    # Convert
    rand_x = rand_r*np.cos(rand_theta)
    rand_y = rand_r*np.sin(rand_theta)
    bot_points = np.column_stack([rand_x, rand_y, rand_z])

    # Generate data for top "hemisphere"
    rand_r = np.cbrt(uniform(radius**3, (radius+width)**3, num_points))
    rand_theta = uniform(0, 2*np.pi, num_points)
    rand_cos_phi = uniform(0, 1, num_points)
    rand_phi = np.arccos(rand_cos_phi) # Need this for uniform distribution
    # Convert
    rand_x = rand_r*np.sin(rand_phi)*np.cos(rand_theta)
    rand_y = rand_r*np.sin(rand_phi)*np.sin(rand_theta)
    rand_z = rand_r*np.cos(rand_phi) + height # Add height to put it on top of cylinder
    top_points = np.column_stack([rand_x, rand_y, rand_z])

    # Create dataframe then save to csv
    all_points = np.vstack([side_points, top_points, bot_points])
    df = pd.DataFrame(
            {"E0": 2447,
            "sx": all_points[:,0],
            "sy": all_points[:,1],
            "sz": all_points[:,2],
            "x": 0,
            "y": 0,
            "z": 0})

    df.to_csv(f"data_emulated/emulated_data_{num_points}points.csv")

def create_shell_df(num_shells, height, radius, mod='quintic'):
    shells = find_shells(num_shells, height/2, radius, mod=mod)
    df = pd.DataFrame({
        "R_shell": shells[1],
        "Z_shell": shells[0]})
    df.to_csv(f"data_emulated/shell_data_{num_shells}shells.csv")

if __name__ == "__main__":
    height=4000
    radius=1500
    width=100
    num_points = 5000000
    num_shells=100

    create_emulated_data(height, radius, width, num_points)
    create_shell_df(num_shells, height, radius)
