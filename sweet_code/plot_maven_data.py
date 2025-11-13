import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from datetime import datetime
import seaborn as sns
from read_maven_sep_data import read_maven_sep_flux_data

# Kan justere størrelse på tekst i figur her
FONTSIZE_AXES_LABELS = 14
FONTSIZE_AXES_TICKS = 12
FONTSIZE_TITLE = 16


def plot_maven_sep_data_channelwise(filename):
    df = read_maven_sep_flux_data(filename) # funksjon fra den andre kodefilen
    # Oppdater current_date til å være midten av tidslinja
    # av plottet du ønsker
    current_date = datetime.strptime("2023-05-10", "%Y-%m-%d")
    start_date = current_date - pd.Timedelta(days=15)
    end_date = current_date + pd.Timedelta(days=15)
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    # df[["datetime", "5820.0-7560.0 keV/n,Ion"]].to_csv("df_maven_attempt_1.csv", sep="\t")

    df_flux_ion = df.iloc[:,3:31] 
    df_flux_electron = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    df_ion = pd.concat([df_datetime, df_flux_ion], axis=1)
    df_electron = pd.concat([df_datetime, df_flux_electron], axis=1)

    df_ion[["datetime", "5820.0-7560.0 keV/n,Ion"]].to_csv("df_maven_attempt_7.csv", sep="\t")

    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    channels = [1, 5, 10, 15, 20, -1], 
    for elem in channels:
        ax1.plot(df_ion["datetime"], df_ion.iloc[:,elem], label=df_ion.columns[elem])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV/n}$"
                   , fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
    + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    ax1.set_xticks(major_ticks_locations)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax1.grid()
    ax1.tick_params(axis="x", rotation=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig.suptitle("",
                  fontsize=FONTSIZE_TITLE)
    
    plt.show()


    fig, ax1 = plt.subplots(figsize=(10, 6))
    channels = [1, 5, 10, 15] 
    for elem in channels:
        ax1.plot(df_electron["datetime"], df_electron.iloc[:,elem], label=df_electron.columns[elem])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV/n}$"
                   , fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
    + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    ax1.set_xticks(major_ticks_locations)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax1.grid()
    ax1.tick_params(axis="x", rotation=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig.suptitle("",
                  fontsize=FONTSIZE_TITLE)
    
    plt.show()



def plot_maven_sep_fluxes_data_heatmap(filename):
    df = read_maven_sep_flux_data(filename)
    # Kan bruke de fire neste linjene med kode for
    # å plotte et tids- intervall av omniweb-dataen
    # Her er et eksempel på 10 dager, der 18. mai 2024 er i midten
    current_date = datetime.strptime("2023-05-05", "%Y-%m-%d")
    start_date = current_date - pd.Timedelta(days=5)
    end_date = current_date + pd.Timedelta(days=5)
    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]


    df_ion_flux = df.iloc[:,3:31]
    df_electron_flux = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    df_ion.set_index('datetime', inplace=True)
    df_electron.set_index('datetime', inplace=True)
    df_ion = df_ion.iloc[:, ::-1]
    df_electron = df_electron.iloc[:, ::-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 20), sharex=True)
    heatmap_ion = sns.heatmap(df_ion.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[0])
    sns.heatmap(df_electron.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[1])
    
    axes[1].legend()
    plt.subplots_adjust(hspace=0.05) 
    cbar = fig.colorbar(heatmap_ion.collections[0], ax=axes, orientation='vertical')
    cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$", 
                   fontsize=FONTSIZE_AXES_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS) 


    axes[0].set_ylabel('Ion energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    axes[1].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    ion_tick_indices = np.arange(0, len(df_ion.index), 24)
    ion_tick_labels = df_ion.index[ion_tick_indices].date  #

    axes[1].set_xticks(ion_tick_indices)
    axes[1].set_xticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS, 
                            rotation=10) 
    axes[0].set_xlabel('')  
    axes[1].set_xlabel('Date', fontsize=FONTSIZE_AXES_LABELS)

    ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]

    ion_axis_ticks = ion_upper_bounds[::3]
    ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) if value in ion_axis_ticks]
    ion_tick_labels = [int(value) if value.is_integer() else value for value in ion_axis_ticks]  
    axes[0].set_yticks(ion_tick_indices)
    axes[0].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)

    # electron_lower_bounds = [float(col.split('-')[0]) for col in df_electron.columns]
    electron_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_electron.columns]
    electron_axis_ticks = electron_upper_bounds[::2]
    electron_tick_indices = [i for i, value in enumerate(electron_upper_bounds) if value in electron_axis_ticks]
    electron_tick_labels =  [int(value) if value.is_integer() else value for value in electron_axis_ticks] 
  
    axes[1].set_yticks(electron_tick_indices)
    axes[1].set_yticklabels(electron_tick_labels, fontsize=FONTSIZE_AXES_TICKS)
    axes[0].tick_params(which='major', axis='y', length=10)  
    axes[0].tick_params(which='minor', axis='y', length=6) 

    axes[1].tick_params(which='major', axis='y', length=10)  
    axes[1].tick_params(which='minor', axis='y', length=6) 

    axes[1].tick_params(which='minor', axis='x', length=6) 
    axes[1].tick_params(which='major', axis='x', length=10) 

    fig.suptitle("MAVEN/SEP hourly fluxes",
                  fontsize=FONTSIZE_TITLE, y=0.95)
    plt.gca().grid(False)
    plt.show()



if __name__ == "__main__":
    filename = 'maven_f_flux_hr_aurora_attempt_2'

    plot_maven_sep_data_channelwise(filename)
    #plot_maven_sep_fluxes_data_heatmap(filename)