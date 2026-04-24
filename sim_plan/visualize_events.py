import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, glob, random
from pathlib import Path

# Functions to help make cylinders
def cylinder(r, h, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def boundary_circle(r, h, nt=100):
    """
    r - boundary circle radius
    h - height above xOy-plane where the circle is included
    returns the circle parameterization
    """
    theta = np.linspace(0, 2*np.pi, nt)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z = h*np.ones(theta.shape)
    return x, y, z


# Choose file from available ones
csv_files = glob.glob("../data/*.csv")

# Load in each file, calculate sr/st/r/t and add a file dictating which file it is
# After all are loaded in, concatenate them into one dataframe
df_list = []
for f in csv_files:
    temp_df = pd.read_csv(f)

    temp_df['sr'] = np.sqrt(temp_df['sx']**2 + temp_df['sy']**2)
    temp_df['st'] = np.atan(temp_df['sy']/temp_df['sx'])
    temp_df['r'] = np.sqrt(temp_df['x']**2 + temp_df['y']**2)
    temp_df['t'] = np.atan(temp_df['y']/temp_df['x'])

    temp_df['filename'] = Path(f).stem
    df_list.append(temp_df)
data = pd.concat(df_list, ignore_index=True)

# Assign some physical values (grabbed from data)
r_max = 1500
z_min = 0
z_max = 4000

# Plotting
fig = go.Figure()

# First plot limited version of the data to reduce lag
limited_data = data.sample(frac=0.01)
colors = [
    "#e41a1c",  # red
    "#ff7f00",  # orange
    "#ffd92f",  # yellow
    "#4daf4a",  # green (not teal)
    "#f781bf",  # pink
    "#984ea3",  # purple
    "#a65628",  # brown
    "#dede00",  # bright yellow-green
    "#fb9a99",  # light red
    "#cab2d6"   # light purple
]
for (fname, group), color in zip(limited_data.groupby('filename'), colors):
    fig.add_trace(go.Scatter3d(
        x=group['x'],
        y=group['y'],
        z=group['z'],
        mode="markers",
        legendgroup=fname,
        name=fname,
        marker=dict(size=3, color=color),
        hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(
        x=group['sx'],
        y=group['sy'],
        z=group['sz'],
        mode='markers',
        legendgroup=fname,
        showlegend=False,
        marker=dict(size=3, color=color, symbol='diamond'),
        hoverinfo='skip'))

# Plot cylinders to view TPC
colorscale = [[0,'blue'],[1,'blue']]
xs, ys, zs = cylinder(r_max, z_max)
fig.add_trace(go.Surface(
    x=xs, y=ys, z=zs,
    colorscale=colorscale,
    showscale=False,
    opacity=0.2,
    name='TPC',
    hoverinfo='skip',
    showlegend=True))

fig.write_html("plot.html")


