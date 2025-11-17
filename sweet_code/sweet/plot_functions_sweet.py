import sys
import os

parent_directory = os.path.abspath('../edac_work')
sys.path.append(parent_directory)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import YearLocator, DayLocator, DateFormatter

from datetime import datetime

from scipy.signal import savgol_filter
from sweet_code.edac.processing_edac import (read_rawedac, 
                             read_zero_set_correct, 
                             read_resampled_df, 
                             read_rolling_rates)

from sweet_code.indices.process_ssn import process_sidc_ssn

from sweet_code.sweet.detrend_edac import read_detrended_rates
from sweet_code.sweet.detect_sw_events import read_sweet_event_dates
from sweet_code.parameters import (LOCAL_DIR,
                                   SWEET_EVENTS_DIR,
                                   TOOLS_OUTPUT_DIR,
                                   SUNSPOTS_SAVGOL,
                                   UPPER_THRESHOLD,
                                   THRESHOLD_COLOR,
                                   RAW_EDAC_COLOR, 
                                   RATE_FIT_COLOR,
                                   SSN_COLOR,
                                   SSN_SMOOTHED_COLOR,
                                   ZEROSET_COLOR,
                                   DETRENDED_EDAC_COLOR,
                                   RATE_EDAC_COLOR,
                                   FONTSIZE_LEGENDS,
                                   FONTSIZE_AXES_LABELS, 
                                   FONTSIZE_AXES_TICKS,
                                   FONTSIZE_TITLE)



def plot_count_rate_with_fit():
    df = read_resampled_df()
    detrended_df = read_detrended_rates()
    rate_mean = round(df['daily_rate'].mean(), 3)
    print(df.sort_values(by='daily_rate'))

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["daily_rate"],
             label=f'MEX EDAC daily rate, mean = {rate_mean} counts per day',
             color=RATE_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax1.plot(
        detrended_df["date"],
        detrended_df["gcr_component"],
        label="Savitzky-Golay fit",
        color=RATE_FIT_COLOR 
    )


    major_y_locator = MultipleLocator(2)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)

    ax1.grid()
    ax1.set_ylim([-1, 20])
    #ax2.set_ylim([0, 18])
    # fig.suptitle("brat")
    fig.suptitle("MEX EDAC count rate",
                 fontsize=FONTSIZE_TITLE-4)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_gcr_fit_ssn():
    detrended_df = read_detrended_rates()
    df_sun = process_sidc_ssn()
    print(detrended_df)
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    # sunspots_smoothed = savgol_filter(df_sun["daily_sunspotnumber"],
    # SUNSPOTS_SAVGOL, 3)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax2.plot(
        detrended_df["date"],
        detrended_df["gcr_component"],
        label="MEX EDAC GCR component",
        color=RAW_EDAC_COLOR
    )

    ax1.plot(
        df_sun["date"],
        df_sun["daily_sunspotnumber"],
        label="Sunspot number",
        color=SSN_COLOR
    )

    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Sunspot number [#]", fontsize=FONTSIZE_AXES_LABELS, color=SSN_COLOR)
    ax2.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS, color=RAW_EDAC_COLOR)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(axis="y", labelcolor=SSN_COLOR) 
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS, 
                    labelcolor=RAW_EDAC_COLOR)
    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)
    major_y_locator = MultipleLocator(50)
    ax1.yaxis.set_major_locator(major_y_locator)

    minor_y_locator = MultipleLocator(25)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    ax1.grid()
    plt.suptitle("EDAC GCR component and the sunspot no. between Jan. 2004 and Aug. 2025",
                 fontsize=FONTSIZE_TITLE)
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="upper left",
        fontsize=FONTSIZE_LEGENDS, bbox_to_anchor=(0.02, 1)) 
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    ax2.set_ylim([detrended_df["gcr_component"].min()-0.05, 2.1])
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_detrended_with_threshold():

    df = read_detrended_rates()
    print(df)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["detrended_rate"],
             label='Detrended count rate',
             color=DETRENDED_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Detrended EDAC count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax1.axhline(
        UPPER_THRESHOLD,
        label="Threshold: " + str(UPPER_THRESHOLD),
        color=THRESHOLD_COLOR
    )

    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    major_y_locator = MultipleLocator(2)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.grid()
    fig.suptitle('MEX EDAC count rates Jan 2004 - Aug 2025',
                 fontsize=FONTSIZE_TITLE,
                 y=0.95)
    plt.show()


def plot_variable_noise_threshold():
    df = read_detrended_rates()

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["detrended_rate"],
             label='Detrended count rate',
             color=DETRENDED_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Detrended count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax1.plot(
        df["date"],
        df["gcr_component"]+1,
        label="Savitzky-Golay fit",
        color=RATE_FIT_COLOR 
    )
    # ax1.axhline(UPPER_THRESHOLD, color='black',
    #            linewidth=2,
    #            linestyle='dashed',
    #            label=f'Threshold of {UPPER_THRESHOLD}')

    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    major_y_locator = MultipleLocator(2)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()
    # fig.suptitle("brat")
    fig.suptitle("MEX EDAC detrended count rate",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()   


def plot_rates_and_ssn():
    """
    Plot the EDAC count rate,
    the standardized count rate,
    and the solar cycle
    for the entire time period covered by
    MEX EDAC
    """
    detrended_df = read_detrended_rates()
    # print("mean: ", detrended_df["detrended_rate"].mean())
    df_sun = process_sidc_ssn()
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    sunspots_smoothed = savgol_filter(df_sun["daily_sunspotnumber"],
                                      SUNSPOTS_SAVGOL, 3)
    # print(detrended_df.sort_values(by="detrended_rate"))
    first_date = detrended_df['date'].iloc[0]
    last_date = detrended_df['date'].iloc[-1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 14))

    ax1.plot(detrended_df["date"],
             detrended_df["daily_rate"], label="EDAC count rate",
             color=RATE_EDAC_COLOR)
    ax1.plot(
        detrended_df["date"],
        detrended_df["gcr_component"],
        label="Savitzky-Golay fit",
        color=RATE_FIT_COLOR
    )
    ax2.plot(
        detrended_df["date"],
        detrended_df["detrended_rate"],
        label="Detrended count rate",
        color=DETRENDED_EDAC_COLOR
    )
    ax3.plot(
        df_sun["date"],
        df_sun["daily_sunspotnumber"],
        label="Sunspot number",
        color=SSN_COLOR
    )
    ax3.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        label="Smoothed sunspot number",
        color=SSN_SMOOTHED_COLOR
    )
    ax3.set_ylabel("Sunspot number [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Detrended count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_xlim([first_date, last_date])
    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)
    ax3.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(100))
    ax3.yaxis.set_minor_locator(MultipleLocator(25))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    ax2.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS) #bbox_to_anchor=(0.99, 1))
    ax3.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)

    plt.subplots_adjust(hspace=0.1)
    fig.suptitle("MEX EDAC count rates with the solar activity cycle",
                 fontsize=FONTSIZE_TITLE, y=0.99) #0.92
    plt.tight_layout(pad=1.0)

    plt.show()


def show_timerange(startdate, enddate):
    """
    For a time period between startdate and enddate,
    create plot of the raw EDAC, the count rate,
    de-trended count rate, standardized count rate
    """
    # raw_edac = read_rawedac()
    raw_edac = read_zero_set_correct()
    filtered_raw = raw_edac.copy()
    filtered_raw = filtered_raw[
        (filtered_raw["datetime"] > startdate) &
        (filtered_raw["datetime"] < enddate)
    ]

    df = read_detrended_rates()
    df = df[(df["date"] > startdate) & (df["date"] < enddate)]
    edac_change = filtered_raw.drop_duplicates(
        subset="edac", keep="first", inplace=False
    )  # Datetimes where the EDAC is increasing

    startdate_string = str(startdate).replace(" ", "_")
    startdate_string = startdate_string.replace(":", "")
    enddate_string = str(enddate).replace(" ", "_")
    enddate_string = enddate_string.replace(":", "")
    # file_name = f"rawEDAC_{startdate_string}-{enddate_string}.txt"

    filtered_raw.loc[:, "time_difference"] = (
        filtered_raw["datetime"].diff().fillna(pd.Timedelta(seconds=0))
    )
    if not os.path.exists(LOCAL_DIR / "events"):
        os.makedirs(LOCAL_DIR / "events")

    filtered_raw.to_csv(
        LOCAL_DIR / "events" /
        f"rawEDAC_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    ) 
    df.to_csv(
        LOCAL_DIR / "events" /
        f"EDACrate_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )  
    edac_change.to_csv(
        LOCAL_DIR / "events" /
        f"EDACrate_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(12, 10))
    ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="MEX EDAC", s=3, color=RAW_EDAC_COLOR)

    ax2.plot(df["date"], df["daily_rate"], marker="o",
             label="EDAC count rate", color=RATE_EDAC_COLOR)

    ax3.plot(df["date"], df["detrended_rate"], marker="o",
             label="Detrended count rate", color=DETRENDED_EDAC_COLOR)

    ax3.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS, labelpad=0)
    ax1.set_ylabel("EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_ylabel("Detrended count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax3.tick_params(axis="x", rotation=10)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax3.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))

    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    ax2.set_ylim([0, 12])
    ax3.set_ylim([-0.5, 10])
    ax3.set_xlim(startdate, enddate)

    major_ticks = pd.date_range(start=startdate+pd.Timedelta(days=1), end=enddate, freq='3D')
    minor_ticks = pd.date_range(start=startdate-pd.Timedelta(days=0), end=enddate, freq='1D')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')
    ax3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')

    fig.suptitle("MEX EDAC counter at the end of May 2006",
                 fontsize=FONTSIZE_TITLE)
    fig.subplots_adjust(top=0.94)

    plt.savefig(LOCAL_DIR / 'events' /
                f'edac_{startdate_string}{enddate_string}.png',
                dpi=300, transparent=False)
    #plt.show()
    plt.close()

    fig, ax1 = plt.subplots(1, figsize=(8, 6))
    ax1.plot(df["date"], df["detrended_rate"], marker="o",
             label="Detrended count rate", color=DETRENDED_EDAC_COLOR)
    ax1.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    
    ax1.axvline(x=datetime.strptime("2006-02-19 12:00", "%Y-%m-%d %H:%M"),
                linestyle='dashed',
                color='black',
                label='SEP event')
    ax2 = ax1.twinx()
    ax2.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="EDAC count", s=3, color=ZEROSET_COLOR)
    
    
    ax1.set_xlabel("Date", fontsize=20),
    ax1.set_ylabel("EDAC count rate [#/day]", fontsize=20, color=DETRENDED_EDAC_COLOR)
    ax2.set_ylabel("EDAC count [#]", fontsize=20, color=ZEROSET_COLOR)
    

    ax1.set_xlim(datetime.strptime("2006-02-13", "%Y-%m-%d"), 
                 datetime.strptime("2006-02-26", "%Y-%m-%d"))
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)


    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))
    date_format = DateFormatter('%b %d')  # '%b' = abbreviated month name, '%d' = day of month
    ax1.xaxis.set_major_formatter(date_format)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles4, labels4 = ax2.get_legend_handles_labels()
    handles = handles1 + handles4
    labels = labels1 + labels4
    ax1.legend(handles, labels, loc='upper left', fontsize=FONTSIZE_LEGENDS)

    #ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')
    ax1.grid()
    fig.suptitle("Example of a SWEET SEP event and FD", fontsize=24, y=1.001)
    plt.tight_layout()
    plt.show()
    #plt.show()
    plt.close()

    fig, ax1 = plt.subplots(1, figsize=(8, 6))
    ax1.plot(df["date"], df["detrended_rate"], marker="o",
             label="Detrended count rate", color=DETRENDED_EDAC_COLOR)
    ax1.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    
    ax1.axvline(x=datetime.strptime("2006-02-19 12:00", "%Y-%m-%d %H:%M"),
                linestyle='dashed',
                color='black',
                label='SEP event')
    ax2 = ax1.twinx()
    ax2.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="EDAC count", s=3, color=ZEROSET_COLOR)
    
    
    ax1.set_xlabel("Date", fontsize=20),
    ax1.set_ylabel("EDAC count rate [#/day]", fontsize=20, color=DETRENDED_EDAC_COLOR)
    ax2.set_ylabel("EDAC count [#]", fontsize=20, color=ZEROSET_COLOR)
    

    ax1.set_xlim(datetime.strptime("2006-02-13", "%Y-%m-%d"), 
                 datetime.strptime("2006-02-26", "%Y-%m-%d"))
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)


    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))
    date_format = DateFormatter('%b %d')  # '%b' = abbreviated month name, '%d' = day of month
    ax1.xaxis.set_major_formatter(date_format)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles4, labels4 = ax2.get_legend_handles_labels()
    handles = handles1 + handles4
    labels = labels1 + labels4
    ax1.legend(handles, labels, loc='upper left', fontsize=FONTSIZE_LEGENDS)

    #ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')
    ax1.grid()
    fig.suptitle("Example of a SWEET SEP event and FD", fontsize=24, y=1.001)
    plt.tight_layout()
    plt.show()


def plot_histogram_rates():
    # Figure for thesis
    """
    Plot histogram distribution
    of detrended EDAC count rate"""
    df = read_detrended_rates()
    data = df["detrended_rate"]
    print(df.sort_values(by='detrended_rate')[-10:])
    mean = df["detrended_rate"].mean()
    std_dev = df['detrended_rate'].std()
    upper_threshold = mean+2*std_dev
    lower_threshold = mean-1.5*std_dev
    # binsize = 0.3
    max_rate = np.max(data)
    min_rate = np.min(data)
    # bins = np.arange(
    #    min_rate, max_rate + binsize, binsize
    # )  # Choose the size of the bins
    bin_edges = np.arange(int(min_rate)-1, int(max_rate)+1, 0.25)
    # print("bin_Edges: ", bin_edges)
    counts, bin_edges = np.histogram(data, bins=bin_edges,
                                     density=False)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bin_edges, color=DETRENDED_EDAC_COLOR,
             edgecolor="black")
    plt.xticks(np.arange(int(min_rate)-2,
                         int(max_rate)+1, 1))
    plt.gca().xaxis.set_minor_locator(
        plt.MultipleLocator(0.5))

    plt.gca().yaxis.set_minor_locator(
        plt.MultipleLocator(100))
    plt.tick_params(which='major', length=10,
                    labelsize=FONTSIZE_AXES_TICKS)
    plt.tick_params(which='minor', length=6,
                    labelsize=FONTSIZE_AXES_TICKS)
    # plt.axvline(x=UPPER_THRESHOLD, label=f'{UPPER_THRESHOLD}',
    # color="#EE7733", linestyle='dashed')
    """
    plt.axvline(x=upper_threshold,
                label=f'Detrended count rate = {upper_threshold}',
                color="#EE7733", linestyle='dashed')

    plt.axvline(x=lower_threshold,
                label=f'Detrended count rate = {lower_threshold}',
                color="#EE7733", linestyle='dashed')
    """

    plt.title("Detrended count rate distribution",
              fontsize=FONTSIZE_TITLE)
    plt.xlabel("Detrended count rate [#/day]",
               fontsize=FONTSIZE_AXES_LABELS)
    plt.ylabel("Occurrences", fontsize=FONTSIZE_AXES_LABELS)
    plt.legend(fontsize=FONTSIZE_LEGENDS)
    plt.xlim(-0.5, 6)
    plt.grid()
    plt.show()


def plot_cumulative_detrended_rates():
    df = read_detrended_rates()
    sorted = np.sort(df["detrended_rate"])
    cdf = np.arange(1, len(sorted) + 1) / len(sorted)
    # print(cdf)
    threshold = np.percentile(sorted, 92)
    print(threshold)
    #percentile = percentileofscore(sorted, 2.3, kind="rank")
    #print(f"The value {2.3} is at the {percentile:.2f}th percentile.")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sorted, cdf, marker='.', 
             linestyle='none', 
             label="CDF of detrended EDAC daily count rates",
             color=DETRENDED_EDAC_COLOR)

    ax1.axhline(
        0.92,
        label="0.92",
        color=THRESHOLD_COLOR
    )

    ax1.set_xlabel("Detrended count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Probability", fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)

    ax1.minorticks_on()

    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.125))

    ax1.legend()
    ax1.grid()
    plt.show()


def create_plots(file_path, date_list, folder_name, event_type_list):
    """
    Create plots of EDAC count,
    EDAC count rate
    and detrended EDAC count rate
    for a given date_list
    """
    if not os.path.exists(file_path / folder_name):
        os.makedirs(file_path / folder_name)
    raw_edac = read_rawedac()
    df = read_detrended_rates()
    df["threshold"] = df["gcr_component"]+1
    count = 0
    for date in date_list:
        print("Date: ", date, ". count: ", count)
        startdate = date - pd.Timedelta(days=21)
        enddate = date + pd.Timedelta(days=21)
        date_string = str(date.date()).replace(" ", "_")
        temp_raw = raw_edac.copy()
        temp_raw = temp_raw[
            (temp_raw["datetime"] > startdate) &
            (temp_raw["datetime"] < enddate)
        ]
        temp_2024 = df.copy()
        temp_2024 = temp_2024[
            (temp_2024["date"] > startdate) &
            (temp_2024["date"] < enddate)
        ]
        vicinity_df = temp_2024[
            (temp_2024["date"] > date - pd.Timedelta(days=2)) &
            (temp_2024["date"] < date + pd.Timedelta(days=2))]
        
 
   
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 7))
        ax1.scatter(temp_raw["datetime"], temp_raw["edac"],
                    label="Raw EDAC", s=3,
                    color=RAW_EDAC_COLOR)
        ax2.plot(
            temp_2024["date"],
            temp_2024["daily_rate"],
            marker="o",
            label="EDAC count rate",
            color=RATE_EDAC_COLOR,
        )

        ax3.plot(
            temp_2024["date"],
            temp_2024["detrended_rate"],
            marker="o",
            label="Detrended rate",
            color=DETRENDED_EDAC_COLOR
        )
        ax3.plot(
            temp_2024["date"],
            temp_2024["threshold"],
            marker="o",
            label="Variable threshold",
            color='red'
        )

        ax3.axhline(UPPER_THRESHOLD, color=THRESHOLD_COLOR, label='Threshold',
                    linestyle='dashed')
        ax3.axvline(x=date, color="black", linewidth="1", label=date)
        ax3.set_xlabel("Date", fontsize=12),
        ax1.set_ylabel("EDAC count", fontsize=12)
        ax2.set_ylabel("EDAC count rate", fontsize=12)
        ax3.set_ylabel("Detrended count rate", fontsize=12)
        ax3.tick_params(axis="x", rotation=20)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.suptitle('December 5th, 2006 SEP event',
        # fontsize=16)
        current_type = event_type_list[count]
        print("current_type: ", current_type)
        # fig.suptitle(f'MSL/RAD SEP start date: {str(date.date())}',
        #              fontsize=16)
        temp_raw = temp_raw[
            (temp_raw["datetime"] > startdate) &
            (temp_raw["datetime"] < enddate)
        ]
        detrended_rate = df.loc[df["date"].dt.date == date.date(), "detrended_rate"].values
        max_detrended_rate = vicinity_df['detrended_rate'].max()
        fig.suptitle(f'{str(date.date())}: {detrended_rate}, {max_detrended_rate}')
        # fig.suptitle(str(date.date()) + ". " + str(current_type),
        # fontsize=16)
        plt.tight_layout(pad=2.0)
        
        plt.savefig(
            file_path / folder_name / f"{str(count)}_{date_string}",
            dpi=300,
            transparent=False,
        )
        # plt.show()
        plt.close()
        count += 1
        

def plot_invalid_edac_increases():
    """
    Make sure to remove days from the list that
    actual have valid increases in addition to the
    invalid ones.
    """
    invalid_dates = pd.read_csv(TOOLS_OUTPUT_DIR / "invalid_edac_increases.txt",
                                parse_dates=["datetime"])

    date_list = invalid_dates["datetime"].tolist()
    event_type_list = [""]*len(date_list)
    folder_name = "invalid_edac_increases"

    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)
    # change invalid_edac_increases_corrected.txt file


def plot_sweet_events_binned_one_plot():
    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)
    # stormy_days_df = read_stormy_sweet_dates()
    # print(f"Number of SWEET stormy days is {len(stormy_days_df)}")
    event_df = read_sweet_event_dates()
    sep_df = event_df[event_df["type"] == "SEP"]
 
    forbush_df = event_df[event_df["type"] == "Fd"]
    print(f'Number of SWEET SEP events: {len(sep_df)}')
    print(f'Number of SWEET Forbush decreases: {len(forbush_df)}')

    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")

    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], SUNSPOTS_SAVGOL, 3
    )

    event_df["6_month_group"] = event_df["date"].apply(group_by_6_months)
    sep_df["6_month_group"] = sep_df["date"].apply(group_by_6_months)
    forbush_df["6_month_group"] = forbush_df["date"].apply(group_by_6_months)

    grouped_df = event_df.groupby("6_month_group").size().reset_index()
    grouped_sep = sep_df.groupby("6_month_group").size().reset_index()
    grouped_fd = forbush_df.groupby("6_month_group").size().reset_index()

    grouped_df.columns = ["datebin", "counts"]
    grouped_sep.columns = ["datebin", "counts"]
    grouped_fd.columns = ["datebin", "counts"]
    grouped_df["datebin"] = grouped_df["datebin"] + pd.DateOffset(months=3)
    grouped_sep["datebin"] = grouped_sep["datebin"] + pd.DateOffset(months=3)
    grouped_fd["datebin"] = grouped_fd["datebin"] + pd.DateOffset(months=3)
    events_total = grouped_df["counts"].sum()
    print("Number of events: ", events_total)


    fig, ax1 = plt.subplots(figsize=(10, 7.5))
    all_events_color = DETRENDED_EDAC_COLOR  # "#8ACE00"
    sepcolor = "#EE3377"  # "#FF6663"
    fdcolor = "#984ea3"
    suncolor = "#9B612F"  # "#a65628"
    ax2 = ax1.twinx()
    ax1.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=2,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax2.plot(
        grouped_df["datebin"],
        grouped_df["counts"],
        marker="o",
        color=all_events_color,
        label="Total number of events",
    )
    ax2.plot(
        grouped_sep["datebin"][:-1],
        grouped_sep["counts"][:-1],
        marker="o",
        color=sepcolor,
        label="Number of SWEET SEP events",
        linewidth=1,
        alpha=1,
    )

    ax2.plot(
        grouped_fd["datebin"],
        grouped_fd["counts"],
        marker="o",
        color=fdcolor,
        label="Number of SWEET FDs",
        linewidth=1,
        alpha=1,
    )

    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.minorticks_on()
    ax1.minorticks_on()
    ax2.xaxis.set_major_locator(YearLocator(4))
    ax2.xaxis.set_minor_locator(YearLocator(1))
    ax2.tick_params(which='minor', length=6)
    ax1.tick_params(which='minor', length=6)

    ax2.set_ylim([0, grouped_df["counts"].max()+10])
    ax1.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Number of SWEET events", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Sunspot number", fontsize=FONTSIZE_AXES_LABELS,
        color=suncolor)
    ax2.tick_params(axis="y") #labelcolor=all_events_color)
    ax1.tick_params(axis="y", labelcolor=suncolor)
    ax2.yaxis.set_minor_locator(MultipleLocator(2))
    ax2.yaxis.set_major_locator(MultipleLocator(4))

    ax1.yaxis.set_minor_locator(MultipleLocator(10))
    ax1.yaxis.set_major_locator(MultipleLocator(20))


    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")


    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Draw grid first with the lowest z-order
    ax2.grid(True, zorder=0)

    ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right',
               bbox_to_anchor=(0.9, 1))
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    # Draw the legend with a higher z-order
    #legend = ax1.legend(
    #    handles + handles2, labels + labels2, 
    #    loc="upper left",
    #    fontsize=FONTSIZE_LEGENDS - 2,
    #    framealpha=1  # Make the legend box opaque
    #)

    # Make sure the legend is on top
    #legend.set_zorder(3)  

    
    #ax2.legend(loc="upper left", fontsize=FONTSIZE_LEGENDS)
    #ax1.legend(loc="upper right", fontsize=FONTSIZE_LEGENDS,
    #           bbox_to_anchor=(0.9, 1))


    plt.title("Sunspot number and number of SWEET events in 6-month bins",
              fontsize=FONTSIZE_TITLE,
              pad=10)
    
    plt.savefig("sweet_binned_v2.png",
    dpi=300,
    transparent=False,
        )
    plt.savefig("sweet_binned_v2.eps",
    dpi=300,
    transparent=False,
        )
    plt.show()


if __name__ == "__main__":
    # plot_count_rate_with_fit()
    # plot_gcr_fit_ssn()
    # plot_rates_and_ssn()
    # plot_histogram_rates()
    # plot_cumulative_detrended_rates()
    plot_detrended_with_threshold()
    plot_sweet_events_binned_one_plot()
    #plot_invalid_edac_increases()