import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from datetime import datetime
import seaborn as sns

from ..parameters import MAVEN_SEP_DIR
from sweet_code.instruments.read_maven_sep_data import read_maven_sep_flux_data

# Kan justere størrelse på tekst i figur her
FONTSIZE_AXES_LABELS = 14
FONTSIZE_AXES_TICKS = 12
FONTSIZE_TITLE = 16


def plot_maven_sep_data_channelwise(current_date):
    
    df = read_maven_sep_flux_data()


    start_date = current_date - pd.Timedelta(days=14)
    start_date = max(start_date, df.iloc[0]["date"])
    end_date = current_date + pd.Timedelta(days=14)
    

    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]
   
    df = df.reset_index(drop=True)
    df_ion_flux = df.iloc[:,3:31]
    df_electron_flux = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    #df_ion.set_index('datetime', inplace=True)
    #df_electron.set_index('datetime', inplace=True)
    df_ion = df_ion.iloc[:, ::-1]
    df_electron = df_electron.iloc[:, ::-1]


    # df_ion[["datetime", "5820.0-7560.0 keV/n,Ion"]].to_csv("df_maven_attempt_7.csv", sep="\t")

    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    channels = [0, 5, 10, 15, 20, 27], 
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
    fig.suptitle(f'{current_date}',
                  fontsize=FONTSIZE_TITLE)
    
    # plt.show()
    path = MAVEN_SEP_DIR / "omniweb/plots"
    plt.savefig(path / f'maven_sep_channelwise_ion_{current_date.date()}.png',
        dpi=300,
        transparent=False)
    


    fig, ax1 = plt.subplots(figsize=(10, 6))
    channels = [0, 5, 10, 14] 
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
    fig.suptitle(f'{current_date}',
                  fontsize=FONTSIZE_TITLE)
    
    # plt.show()


    plt.savefig(path / f'maven_sep_channelwise_electron_{current_date.date()}.png',
        dpi=300,
        transparent=False)
    

def plot_maven_sep_fluxes_data_heatmap(current_date):

    df = read_maven_sep_flux_data()


    start_date = current_date - pd.Timedelta(days=14)
    start_date = max(start_date, df.iloc[0]["date"])
    end_date = current_date + pd.Timedelta(days=14)
    

    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]
   
    df = df.reset_index(drop=True)
    df_ion_flux = df.iloc[:,3:31]
    df_electron_flux = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    df_ion.set_index('datetime', inplace=True)
    df_electron.set_index('datetime', inplace=True)
    df_ion = df_ion.iloc[:, ::-1]
    df_electron = df_electron.iloc[:, ::-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
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

    # ADJUST THE TICKS HERE
    tick_index_list = []
    tick_label_list = []
    interval_length = end_date - start_date

    if interval_length >= pd.Timedelta(days=24):
        days_interval_ticks = 5
    else:
        days_interval_ticks = 4
    tick_date_range = pd.date_range(current_date-pd.Timedelta(days=2*days_interval_ticks), 
                                    current_date+pd.Timedelta(days=2*days_interval_ticks),
                                    freq=f'{days_interval_ticks}D')

   
    for tick_date in tick_date_range:
        midnight_tick_date = datetime.combine(tick_date, datetime.min.time())
        midnight_tick_index = df.index[df['datetime'] == midnight_tick_date].tolist()[0]
        tick_label_list.append(midnight_tick_date.date())
        tick_index_list.append(midnight_tick_index)



    ion_tick_labels = tick_label_list # df_ion.index[tick_index_].date  
    ion_tick_indices= tick_index_list


    axes[1].set_xticks(ion_tick_indices)
    axes[1].set_xticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS, 
                            rotation=10) 

    minor_tick_index_list = []
    minor_tick_date_range =  pd.date_range(start_date, end_date,freq='D')
    """
    for minor_tick_date in minor_tick_date_range():
        midnight_tick_date = datetime.combine(minor_tick_date, datetime.min.time())
        midnight_tick_index = df.index[df['datetime'] == midnight_tick_date].tolist()[0]
        minor_tick_index_list.append(midnight_tick_index)
    """
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

    fig.suptitle(f'MAVEN/SEP hourly flux around {current_date.date()}',
                  fontsize=FONTSIZE_TITLE, y=0.92)
    plt.gca().grid(False)
    # plt.show()

    path = MAVEN_SEP_DIR / "omniweb/plots"

    plt.savefig(path / f'maven_sep_{current_date.date()}.png',
        dpi=300,
        transparent=False)



def plot_series_of_maven_sep_heatmaps():
    """
    Create MAVEN/SEP heatmaps for a list of dates

    """
    date_list = [datetime.strptime("2024-05-20", "%Y-%m-%d"),
                 datetime.strptime("2017-09-10", "%Y-%m-%d"),
                 datetime.strptime("2023-08-31", "%Y-%m-%d"),
                 datetime.strptime("2024-03-17", "%Y-%m-%d"),
                 datetime.strptime("2024-07-23", "%Y-%m-%d"),
                 datetime.strptime("2023-04-03", "%Y-%m-%d"),
                 datetime.strptime("2023-07-01", "%Y-%m-%d"),
                 datetime.strptime("2023-06-13", "%Y-%m-%d"),
                 datetime.strptime("2014-10-01", "%Y-%m-%d"),
                 datetime.strptime("2023-05-06", "%Y-%m-%d"),
                 datetime.strptime("2022-12-27", "%Y-%m-%d"),
                 datetime.strptime("2016-01-15", "%Y-%m-%d"),
                 datetime.strptime("2023-12-05", "%Y-%m-%d"),
                 datetime.strptime("2018-03-24", "%Y-%m-%d"),
                 datetime.strptime("2017-02-22", "%Y-%m-%d")]

    for date in date_list:
        print(date)
        plot_maven_sep_fluxes_data_heatmap(date)

def plot_series_of_maven_sep_channels():
    date_list = [datetime.strptime("2024-05-20", "%Y-%m-%d"),
                 datetime.strptime("2017-09-10", "%Y-%m-%d"),
                 datetime.strptime("2023-08-31", "%Y-%m-%d"),
                 datetime.strptime("2024-03-17", "%Y-%m-%d"),
                 datetime.strptime("2024-07-23", "%Y-%m-%d"),
                 datetime.strptime("2023-04-03", "%Y-%m-%d"),
                 datetime.strptime("2023-07-01", "%Y-%m-%d"),
                 datetime.strptime("2023-06-13", "%Y-%m-%d"),
                 datetime.strptime("2014-10-01", "%Y-%m-%d"),
                 datetime.strptime("2023-05-06", "%Y-%m-%d"),
                 datetime.strptime("2022-12-27", "%Y-%m-%d"),
                 datetime.strptime("2016-01-15", "%Y-%m-%d"),
                 datetime.strptime("2023-12-05", "%Y-%m-%d"),
                 datetime.strptime("2018-03-24", "%Y-%m-%d"),
                 datetime.strptime("2017-02-22", "%Y-%m-%d")]

    for date in date_list:
        print(date)
        plot_maven_sep_data_channelwise(date)


if __name__ == "__main__":
    # plot_series_of_maven_sep_heatmaps()

    # current_date = datetime.strptime("2024-05-20", "%Y-%m-%d")
    # plot_maven_sep_data_channelwise(current_date)
    # plot_maven_sep_fluxes_data_heatmap(current_date)
    # plot_maven_sep_fluxes_data_heatmap(filename)
    plot_series_of_maven_sep_channels()