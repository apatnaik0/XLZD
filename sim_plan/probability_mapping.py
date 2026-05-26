import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from visualize_events import create_plot
from scipy.stats import binned_statistic_2d

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

max_z = np.max(data['z'])
min_z = np.min(data['z'])
max_r = np.max(data['r'])
center_z = (min_z+max_z)/2

nx=100
ny=100

# Top
top_data = data[(data['sr']<max_r) & (data['sz']>max_z)]
create_plot(top_data, plot_name="TopData.html", r_max=max_r, z_min=min_z, z_max=max_z)

x_top = top_data['sx'].to_numpy()
y_top = top_data['sy'].to_numpy()
z_top = np.sqrt(top_data['r']**2 + (top_data['z']-center_z)**2).to_numpy()
avgs_top, x_edge_top, y_edge_top, binnumber = binned_statistic_2d(
        x_top,y_top,z_top, 
        statistic='mean', bins=[nx,ny])

# Bottom
bot_data = data[(data['sr']<max_r) & (data['sz']<max_z)]
create_plot(bot_data, plot_name="BotData.html", r_max=max_r, z_min=min_z, z_max=max_z)

x_bot = bot_data['sx'].to_numpy()
y_bot = bot_data['sy'].to_numpy()
z_bot = np.sqrt(bot_data['r']**2 + (bot_data['z']-center_z)**2).to_numpy()
avgs_bot, x_edge_bot, y_edge_bot, binnumber = binned_statistic_2d(
        x_bot,y_bot,z_bot, 
        statistic='mean', bins=[nx,ny])

# Side
side_data = data[(data['sr']>max_r) & (min_z<data['sz']) & (data['sz']<max_z)]
create_plot(side_data, plot_name="SideData.html", r_max=max_r, z_min=min_z, z_max=max_z)

x_side = side_data['st'].to_numpy()
y_side = side_data['sz'].to_numpy()
z_side = np.sqrt(side_data['r']**2 + (side_data['z']-center_z)**2).to_numpy()
avgs_side, x_edge_side, y_edge_side, binnumber = binned_statistic_2d(
        x_side,y_side,z_side, 
        statistic='mean', bins=[nx,ny])

# Put together plot
fig, axs = plt.subplots(1,3,figsize=(13,5),constrained_layout=True)

vmin = np.nanmin([avgs_top, avgs_bot, avgs_side])
vmax = np.nanmax([avgs_top, avgs_bot, avgs_side])

im0 = axs[0].pcolormesh(x_edge_top, y_edge_top, avgs_top.T, vmin=vmin, vmax=vmax)
im1 = axs[1].pcolormesh(x_edge_bot, y_edge_bot, avgs_bot.T, vmin=vmin, vmax=vmax)
im2 = axs[2].pcolormesh(x_edge_side, y_edge_side, avgs_side.T)

axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].set_title("Distance from Center of Top Sources")

axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].set_title("Distance from Center of Bottom Sources")

axs[2].set_xlabel("Theta")
axs[2].set_ylabel("Z")
axs[2].set_title("Distance from Center of Side Sources")

cbar = fig.colorbar(im0, ax=axs, orientation='vertical')
cbar.set_label('Average Distance from Center')
fig.savefig("ProbabilityPlot.png")
