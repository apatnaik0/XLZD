import glob
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def find_shells(num_shells, z_max, r_max, mod="const_vol"):
    int_array = np.arange(num_shells)+1
    ratio = int_array/num_shells
    if mod == "const_vol":
        modifier = np.cbrt(ratio)
    elif mod == "linear":
        modifier = ratio
    elif mod == "exp":
        modifier = np.exp(ratio-1)
    elif mod == "quintic":
        modifier = np.power(ratio, 1/5)
    else:
        raise ValueError(f"Modifier {mod} not allowed")
    shell_z = z_max*modifier
    shell_r = r_max*modifier

    return (shell_z, shell_r)

def df_shell_occupations(df, z_shells, r_shells):
    # Add in a 0 so all shell edges are counted
    z_shells = np.insert(z_shells,0,0)
    r_shells = np.insert(r_shells,0,0)
    
    # Iterate through each shell and fill out the dataframe with a new column
    df = df.copy()
    for shell_num, (z, r) in enumerate(zip(z_shells[1:], r_shells[1:]), start=1):
        # Setup Ranges
        z_max = z
        z_min = z_shells[shell_num-1]
        r_max = r
        r_min = r_shells[shell_num-1]
        
        # Create Conditions - One for wall of cylinder shell, one for cylinder caps
        wall_condition = ( 
                (np.abs(df['z']-z_shells[-1]) < z_max) & # Within Z range 
                (r_min < df['r']) & (df['r'] < r_max) # Within R Shell
            )
        cap_condition = (
                (df['r'] < r_max) & # Within R range
                (z_min < np.abs(df['z']-z_shells[-1])) & (np.abs(df['z']-z_shells[-1]) < z_max) # Within Z shell
            )
        condition = wall_condition | cap_condition

        # Finally log shell number
        df.loc[condition, "shell_num"] = shell_num
    return df

def find_shell_occupation(df, z_shells, r_shells, num_shells):
    # Fill all the shell numbers then count each integer and arrange them
    df = df_shell_occupations(df, z_shells, r_shells)
    shell_nums = df["shell_num"]
    occupation = (shell_nums.value_counts().reindex(range(1, num_shells+1), fill_value=0).to_numpy())

    return occupation

def gauss(x, a, s):
    return a*np.exp(s*x)

def occupation_regression(occ, num_shells):
    x_data = np.arange(num_shells)+1
    p0 = (.1,.2)
    popt, pcov = curve_fit(gauss, x_data, occ, p0, maxfev=10000)
    return (popt[0], popt[1])

def shell_occupancy(data, modifiers, max_z, max_r, num_shells=100):
    # Run through different modifiers
    for mod in modifiers:
        print(f"Starting run with modifier {mod}")

        # Setup shells and find occupation num
        x_data = np.arange(num_shells)+1
        zs, rs = find_shells(num_shells, max_z, max_r, mod=mod)
        occ = find_shell_occupation(data, zs, rs, num_shells)
        
        # Grab regression of plot
        a, s = occupation_regression(occ, num_shells)
        regression_func = gauss(x_data, a, s)

        # Plot
        min_occ, min_idx = min([(v,i) for i, v in enumerate(occ) if v>0])

        plt.bar(x_data, occ)
        plt.plot(x_data, regression_func, color='red')
        plt.grid()
        plt.xlabel("Shell Number")
        plt.ylabel("Count")
        plt.title(f"Number of Events in each Shell\nUsing {mod} modifier")
        plt.annotate(f"{min_occ} Occupation\nin Shell {min_idx+1}", xy=(min_idx, min_occ), xytext=(20,10), textcoords='offset points', ha='center', va='bottom')
        plt.annotate(f"Regresion: {round(a,4)}*e^({round(s,4)}x)", xy=(0.05, 0.90), xycoords='axes fraction')
        plt.tight_layout()
        plt.savefig(f'results/ShellCount_{mod}.png')
        plt.clf()

def shell_distribution(data, max_z, maz_r, num_shells):
    # First take the data and only keep 0<sz<max_z
    data = data[(data['sz']<max_z*2) & (0<data['sz'])]

    # Find shell numbers of each of the points
    shell_z, shell_r = find_shells(num_shells, max_z, max_r, mod="quintic")
    df = df_shell_occupations(data, shell_z, shell_r)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    plot = ax.scatter(df['sx'], df['sy'], df['sz'], c=df['shell_num'], s=0.5, alpha=0.2)
    cbar = plt.colorbar(plot)
    cbar.set_label('Shell Number')
    plt.tight_layout()
    plt.savefig("results/TestPlot.png")

if __name__ == "__main__":
    # Choose file from available ones
    data_file = "../data/*.csv"
    csv_files = glob.glob(data_file)

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

    # Calculate detector size from data
    max_z = np.ceil(max(data['z'])/2)
    max_r = np.ceil(max(data['r']))

    num_shells = 100
    #shell_occupancy(data, ["const_vol", "quintic", "linear", "exp"], max_z, max_r, num_shells)
    #shell_distribution(data, max_z, max_r, num_shells)
