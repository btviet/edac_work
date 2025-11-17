import sys
import os

parent_directory = os.path.abspath('../edac_work')
sys.path.append(parent_directory)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import YearLocator
from datetime import datetime
from sweet_code.edac.processing_edac import read_rawedac
from sweet_code.sweet.detrend_edac import read_detrended_rates
from sweet_code.instruments.read_msl_rad_data import read_msl_rad_doses
from sweet_code.parameters import (LOCAL_DIR,
                                   RAW_EDAC_COLOR,
                                   ZEROSET_COLOR,
                                   DETRENDED_EDAC_COLOR,
                                   UPPER_THRESHOLD,
                                   THRESHOLD_COLOR,
                                   LOWER_THRESHOLD,
                                   DETRENDED_EDAC_COLOR,
                                   RAD_B_COLOR,
                                   RAD_E_COLOR,
                                   FONTSIZE_LEGENDS,
                                   FONTSIZE_AXES_LABELS, 
                                   FONTSIZE_AXES_TICKS,
                                   FONTSIZE_TITLE)


def plot_msl_rad_all():
    df = read_msl_rad_doses()
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df['datetime'], df['B_dose'], label='B dose rate',
             color=RAD_B_COLOR)
    ax1.plot(df['datetime'], df['E_dose'], label='E dose rate', linestyle='dashed',
             color=RAD_E_COLOR)
    
    ax1.xaxis.set_major_locator(YearLocator(2))
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator())
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylim(-1, 2)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()
    fig.suptitle("MSL/RAD dose rates from August 2012 to Sept 2025",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.close()
    #plt.show()


    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 7))
    ax1.plot(df['datetime'], df['B_dose'], label='B dose rate',
             color=RAD_B_COLOR)
    ax1.plot(df['datetime'], df['E_dose'], label='E dose rate', linestyle='dashed',
             color=RAD_E_COLOR)
    
    ax1.xaxis.set_major_locator(YearLocator(2))
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator())
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    # ax1.set_ylim(-1, 2)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()

    ax2.plot(df['datetime'], df['B_dose'], label='B dose rate',
             color=RAD_B_COLOR)
    ax2.plot(df['datetime'], df['E_dose'], label='E dose rate', linestyle='dashed',
             color=RAD_E_COLOR)
    
    ax2.xaxis.set_major_locator(YearLocator(2))
    ax2.xaxis.set_minor_locator(YearLocator(1))
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylim(-0.01, 1.1)
    ax1.set_ylim(0, 9)
    ax2.legend(fontsize=FONTSIZE_LEGENDS)
    ax2.grid()
    lower_xlim = df['datetime'].iloc[0] - pd.Timedelta(days=180)
    higher_xlim = df['datetime'].iloc[-1] + pd.Timedelta(days=180)
    ax2.set_xlim(lower_xlim, higher_xlim)


    fig.suptitle("MSL/RAD dose rates from August 2012 to September 2025",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    
    plt.subplots_adjust(hspace=0.2)   
    #plt.savefig('msl_rad_all_v4.eps',
    #            dpi=300, transparent=False,
    #             bbox_inches="tight", pad_inches=0.05)
    #plt.savefig('msl_rad_all_v4.png',
    #            dpi=300, transparent=False,
    #             bbox_inches="tight", pad_inches=0.05)
    
    plt.show()



def plot_rad_sweet_samplewise(start_time, end_time, onset_time):
    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_time) & (df_raw["datetime"] <= end_time)]
    df_rad = read_msl_rad_doses()
    df_rad = df_rad[(df_rad["datetime"] >= start_time) & (df_rad["datetime"] <= end_time)]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                        figsize=(10, 6))
    ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='B dose rate', marker='o',
             markersize=5, color=RAD_B_COLOR)
    ax1.plot(df_rad['datetime'], df_rad['E_dose'], label='E dose rate', marker='o',
             markersize=5, color=RAD_E_COLOR)

    
    ax2.scatter(df_raw["datetime"], df_raw["edac"],
            label='MEX EDAC counter value',
            color=RAW_EDAC_COLOR,
            marker='o',
            s=5)

    ax1.axvline(x=onset_time,
                linestyle='dashed',
                color='black',
                label='Onset time of RAD event')
    ax2.axvline(x=onset_time,
                linestyle='dashed',
                color='black',
                label='Onset time of RAD event')
    
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.grid()
    ax2.grid()

    ax1.legend()
    ax2.legend()
    plt.show()
    #if not os.path.exists(LOCAL_DIR / "events/sweet_rad_comparison"):
    #    os.makedirs(LOCAL_DIR / "events/sweet_rad_comparison")

    #plt.savefig(LOCAL_DIR / 'events/sweet_rad_comparison' /
    #            f'timing_sweet_rad_{str(onset_time.date())}.png',
    #            dpi=300, transparent=False)
    #plt.close()



def plot_msl_rad_sweet(start_date, end_date):
    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_date) & (df_raw["datetime"] <= end_date)]
    df_rad = read_msl_rad_doses()
    df_rad = df_rad[(df_rad["datetime"] >= start_date) & (df_rad["datetime"] <= end_date)]
    df_sweet = read_detrended_rates()
    df_sweet = df_sweet[(df_sweet["date"] > start_date) & (df_sweet["date"] < end_date)]

    fig, ax1 = plt.subplots(1, figsize=(10, 6))
    ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='RAD B dose rate', marker='o',
             markersize=2, color=RAD_B_COLOR)
    
    ax2 = ax1.twinx()
    ax2.scatter(df_raw["datetime"], df_raw["edac"],
            label='MEX EDAC count',
            color= ZEROSET_COLOR,# RAW_EDAC_COLOR,
            marker='o',
            s=5)
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS-2, color=ZEROSET_COLOR)
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS-2, color=RAD_B_COLOR)
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    ax2.xaxis.set_minor_locator(mdates.DayLocator())
    major_ticks_locations =  [pd.to_datetime('2013-04-11 00:00:00') + pd.Timedelta(days=2 * i) 
                              for i in range(-3, 3)]  
    ax2.set_xticks(major_ticks_locations)

    date_format = mdates.DateFormatter('%b %d')  # '%b' = abbreviated month name, '%d' = day of month
    ax1.xaxis.set_major_formatter(date_format)

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles4, labels4 = ax2.get_legend_handles_labels()
    handles = handles1 + handles4
    labels = labels1 + labels4
    ax1.legend(handles, labels, loc='upper left', fontsize=FONTSIZE_LEGENDS)
    #ax1.set_xlim(datetime.strptime("2024-05-18", "%Y-%m-%d"), 
    #             datetime.strptime("2024-05-24", "%Y-%m-%d"))
    ax1.grid()

    fig.suptitle("MSL/RAD and MEX EDAC during May 2024 event", fontsize=FONTSIZE_TITLE)
    plt.tight_layout()
    plt.show()

    ##
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(10, 6))
    
    print(df_rad)
    ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='B dose rate', marker='o',
             markersize=2, color=RAD_B_COLOR)
    ax1.plot(df_rad['datetime'], df_rad['E_dose'], label='E dose rate', marker='o',
             markersize=2, color=RAD_E_COLOR)
    #ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='B dose rate', 
    #         color=RAD_B_COLOR)
    #ax1.plot(df_rad['datetime'], df_rad['E_dose'], label='E dose rate',
    #          color=RAD_E_COLOR)
    
    ax2.scatter(df_raw["datetime"], df_raw["edac"],
            label='MEX EDAC counter value',
            color= ZEROSET_COLOR,# RAW_EDAC_COLOR,
            marker='o',
            s=5)
    
    ax3.plot(df_sweet["date"], df_sweet["detrended_rate"],
             marker='o',
             color=DETRENDED_EDAC_COLOR,
             label='Detrended EDAC count rate')
    
    ax3.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    
    #ax1.axvline(x=datetime.strptime("2015-05-05 10:00:00", "%Y-%m-%d %H:%M:%S"),
    #            linestyle='dashed',
    #            color='black',
    #            label='Onset of RAD FD')
    
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS-2)
    ax3.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS-2)
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS-2)
    
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))
    date_format = mdates.DateFormatter('%b %d')  # '%b' = abbreviated month name, '%d' = day of month
    ax1.xaxis.set_major_formatter(date_format)
    """
    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
        + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]

    ax3.set_xticks(major_ticks_locations)
    """
    #ax3.xaxis.set_major_locator(mdates.DayLocator(3))
    ax3.xaxis.set_minor_locator(mdates.DayLocator())
    major_ticks_locations =  [pd.to_datetime('2013-04-11 00:00:00') + pd.Timedelta(days=2 * i) 
                              for i in range(-2, 4)]  
    ax3.set_xticks(major_ticks_locations)

    lower_xlim = df_sweet['date'].iloc[0] - pd.Timedelta(days=0)
    higher_xlim = df_sweet['date'].iloc[-1] + pd.Timedelta(days=0)
    ax3.set_xlim(lower_xlim, higher_xlim)
    #ax3.set_ylim(-1, 4.2)
    #ax1.set_ylim(0.17, 0.3)
    
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(loc='upper left')
    ax2.legend()
    ax3.legend()
    fig.suptitle("MSL/RAD and SWEET during May 2024 event",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=0.5)

    
    if not os.path.exists(LOCAL_DIR / "events/sweet_rad_comparison"):
        os.makedirs(LOCAL_DIR / "events/sweet_rad_comparison")

    plt.savefig(LOCAL_DIR / 'events/sweet_rad_comparison' /
                f'sweet_rad_{str(start_date.date())}.png',
                dpi=300, transparent=False)
    print("figure created")
    plt.show()

    df_rad['time_difference'] = df_rad['datetime'].diff()
    df_rad['time_difference_in_minutes'] = \
        df_rad['time_difference'].dt.total_seconds() / 60
    
    print("Max time difference: ", 
          df_rad['time_difference_in_minutes'].max())
    df_rad.to_csv(LOCAL_DIR / 'events/sweet_rad_comparison'/ "rad_data_temp.txt")
    # df_rad.to_csv("test_rad.txt")

    row_B= df_rad[(df_rad["B_dose"] == df_rad['B_dose'].max())]
    print(row_B[['datetime', 'B_dose', 'B_dose_err']])
    row_E= df_rad[(df_rad["E_dose"] == df_rad['E_dose'].max())]
    print(row_E[['datetime', 'E_dose', 'E_dose_err']])



def plot_rad_sweet_rate_and_count(onset_time):
    start_date = onset_time - pd.Timedelta(days=4)
    end_date = onset_time + pd.Timedelta(days=7)
    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_date) & (df_raw["datetime"] <= end_date)]
    df_rad = read_msl_rad_doses()
    df_rad = df_rad[(df_rad["datetime"] >= start_date) & (df_rad["datetime"] <= end_date)]
    df_sweet = read_detrended_rates()
    df_sweet = df_sweet[(df_sweet["date"] > start_date) & (df_sweet["date"] < end_date)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(10, 6))
    
    ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='B dose rate', marker='o',
             markersize=2, color=RAD_B_COLOR)
    ax1.plot(df_rad['datetime'], df_rad['E_dose'], label='E dose rate', marker='o',
             markersize=2, color=RAD_E_COLOR)
    #ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='B dose rate', 
    #         color=RAD_B_COLOR)
    #ax1.plot(df_rad['datetime'], df_rad['E_dose'], label='E dose rate',
    #          color=RAD_E_COLOR)
    
    ax2.scatter(df_raw["datetime"], df_raw["edac"],
            label='MEX EDAC counter value',
            color= ZEROSET_COLOR,# RAW_EDAC_COLOR,
            marker='o',
            s=5)
    
    ax3.plot(df_sweet["date"], df_sweet["detrended_rate"],
             marker='o',
             color=DETRENDED_EDAC_COLOR,
             label='Detrended EDAC count rate')
    
    ax3.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    
    #ax1.axvline(x=datetime.strptime("2015-05-05 10:00:00", "%Y-%m-%d %H:%M:%S"),
    #            linestyle='dashed',
    #            color='black',
    #            label='Onset of RAD FD')
    
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS-4)
    ax3.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS-4)
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS-4)
    
    ax1.yaxis.set_major_locator(MultipleLocator(0.1)) # dose rate
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05)) # dose rate

    ax2.yaxis.set_major_locator(MultipleLocator(4)) # edac count
    ax2.yaxis.set_minor_locator(MultipleLocator(2)) # edac count
    ax3.yaxis.set_major_locator(MultipleLocator(2)) # edac count rate
    ax3.yaxis.set_minor_locator(MultipleLocator(1)) # edac count rate
    date_format = mdates.DateFormatter('%b %d')  # '%b' = abbreviated month name, '%d' = day of month
    ax1.xaxis.set_major_formatter(date_format)
    #ax3.xaxis.set_major_locator(mdates.DayLocator(3))
    ax3.xaxis.set_minor_locator(mdates.DayLocator())
    major_ticks_locations =  [onset_time.date() + pd.Timedelta(days=2 * i) 
                              for i in range(-2, 4)]  
    ax3.set_xticks(major_ticks_locations)

    lower_xlim = df_sweet['date'].iloc[0] - pd.Timedelta(days=0)
    higher_xlim = df_sweet['date'].iloc[-1] + pd.Timedelta(days=0)
    ax3.set_xlim(lower_xlim, higher_xlim)
    ax3.set_ylim(-1, 10)
    #ax1.set_ylim(0.17, 0.3)
    
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(loc='upper left')
    ax2.legend()
    ax3.legend()
    fig.suptitle("MSL/RAD and MEX EDAC",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=0.5)

    """
    if not os.path.exists(LOCAL_DIR / "events/sweet_rad_comparison"):
        os.makedirs(LOCAL_DIR / "events/sweet_rad_comparison")

    plt.savefig(LOCAL_DIR / 'events/sweet_rad_comparison' /
                f'sweet_rad_{str(start_date.date())}.png',
                dpi=300, transparent=False)
    print("figure created")
    """
    plt.show()



if __name__ == "__main__":
    # start_time = datetime.strptime("2013-07-17", "%Y-%m-%d %H:%M")
    onset_time = datetime.strptime("2024-07-23 17:48", "%Y-%m-%d %H:%M")
    start_time = onset_time - pd.Timedelta(days=4)
    end_time = onset_time + pd.Timedelta(days=3)
    # plot_rad_sweet_samplewise(start_time, end_time, onset_time)
    #plot_msl_rad_sweet(start_time, end_time)
    plot_rad_sweet_rate_and_count(onset_time)