import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detect_sw_events import (
    read_extra_sweet_sep_events,
    read_stormy_sweet_dates,
    read_sweet_event_dates,
    read_sweet_forbush_decreases,
    read_sweet_sep_events,
)
from detrend_edac import read_detrended_rates
from matplotlib.dates import YearLocator, DayLocator
from matplotlib.ticker import MultipleLocator
from old_sweet_comparison import read_cme_validation_results_old
from parameters import (
    IMA_COLOR,
    BRAT_GREEN,
    DETRENDED_EDAC_COLOR,
    FONTSIZE_AXES_LABELS,
    FONTSIZE_AXES_TICKS,
    FONTSIZE_LEGENDS,
    FONTSIZE_TITLE,
    LOCAL_DIR,
    RATE_EDAC_COLOR,
    RATE_FIT_COLOR,
    RAW_DATA_DIR,
    RAW_EDAC_COLOR,
    SEP_VALIDATION_DIR,
    SSN_COLOR,
    SSN_SMOOTHED_COLOR,
    STANDARDIZED_EDAC_COLOR,
    SUNSPOTS_SAVGOL,
    SWEET_EVENTS_DIR,
    THRESHOLD_COLOR,
    UPPER_THRESHOLD,
    ZEROSET_COLOR,
)
from processing_edac import (
    read_rawedac,
    read_resampled_df,
    read_rolling_rates,
    read_zero_set_correct,
)
from read_from_database import (
    read_forbush_decreases_maven,
    read_forbush_decreases_rad,
    read_sep_events_maven,
    read_sep_events_rad,
    read_mex_safe_modes,
)
from scipy.signal import savgol_filter
from validate_cme_events import read_cme_validation_results
from validate_forbush_decreases import read_msl_rad_fd_validation_result
from validate_sep_events import read_sep_validation_results
from read_mex_aspera_data import (
    read_mex_ima_bg_counts, 
    clean_up_mex_ima_bg_counts,
    read_aspera_sw_moments)



def plot_raw_edac_scatter(start_date, end_date):
    """
    For small time intervals
    """
    df = read_rawedac()
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    print("df: ", df)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(df["datetime"], df["edac"],
                label='Raw MEX EDAC',
                color=RAW_EDAC_COLOR,
                linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.tick_params(axis="x", rotation=10)
    major_ticks = pd.date_range(start=start_date, end=end_date, freq='2D')
    ax1.set_xticks(major_ticks)

    minor_ticks = pd.date_range(start=start_date, end=end_date, freq='D')
    ax1.set_xticks(minor_ticks, minor=True)
    """
    major_ticks_locations = [
        pd.to_datetime('2024-05-17 12:00:00')
        + pd.Timedelta(days=2 * i)
        for i in range(-1, 7)]
    ax1.set_xticks(major_ticks_locations)
    
    minor_ticks_locations = [
        pd.to_datetime('2024-05-17 12:00:00') + pd.Timedelta(hours=i)
        for i in range(0, 48 * 2)  # 48 hours * 2 (to cover minor ticks for 2 days)
    ]
    """
    # ax1.set_xticks(minor_ticks_locations, minor=True)
    # major_x_locator = YearLocator(4)
    # ax1.xaxis.set_major_locator(major_x_locator)
    # ax1.minorticks_on()
    # minor_x_locator = YearLocator(2)
    # ax1.xaxis.set_minor_locator(minor_x_locator)

    # major_y_locator = MultipleLocator(5000)
    # ax1.yaxis.set_major_locator(major_y_locator)
    # minor_y_locator = MultipleLocator(2500)
    # ax1.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend()
    ax1.grid()
    fig.suptitle("MEX EDAC counter",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_raw_edac():
    # Figure in thesis
    df = read_rawedac()
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["datetime"], df["edac"],
             label='Raw MEX EDAC',
             color=RAW_EDAC_COLOR,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(2)
    ax1.xaxis.set_minor_locator(minor_x_locator)
    major_y_locator = MultipleLocator(5000)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(2500)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend()
    ax1.grid()
    fig.suptitle("brat")
    # fig.suptitle("MEX EDAC counter ..",
    # fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_zero_set_correction():
    # Figure in thesis
    df = read_zero_set_correct()
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["datetime"], df["edac"],
             label='MEX EDAC counter',
             color=RAW_EDAC_COLOR,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)
    major_y_locator = MultipleLocator(2000)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(500)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()
    fig.suptitle("EDAC counter between Jan 2004 and Apr 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_raw_and_zerosetcorrected():
    """
    Plot the raw EDAC counter
    and the zeroset-corrected counter
    in one plot.
    """
    df = read_rawedac()
    df_zero = read_zero_set_correct()

    currentdate = datetime.strptime("2006-01-06", "%Y-%m-%d")
    startdate = currentdate - pd.Timedelta(days=7)
    enddate = currentdate + pd.Timedelta(days=7)

    df = df[
        (df["datetime"] > startdate) &
        (df["datetime"] < enddate)
        ]
    print(df)
    df_zero = df_zero[(df_zero["datetime"] > startdate) &
    (df_zero["datetime"] < enddate)
    ]
    

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(df["datetime"], df["edac"],
             label='MEX EDAC',
             color=RAW_EDAC_COLOR,
             linewidth=2.5)
    ax2 = ax1.twinx()
    ax2.plot(df_zero["datetime"], df_zero["edac"],
             label='Zero-set corrected MEX EDAC',
             color=ZEROSET_COLOR,
             linewidth=2.5)
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count [#]",
                   fontsize=FONTSIZE_AXES_LABELS,
                   color='#00456c')  # RAW_EDAC_COLOR)
    ax2.set_ylabel("Zero-set corrected MEX EDAC count [#]",
                   fontsize=FONTSIZE_AXES_LABELS, color='#b93819',
                   labelpad=10)  # ZEROSET_COLOR,
    ax1.tick_params(axis="y", labelcolor='#00456c')  # RAW_EDAC_COLOR)
    ax2.tick_params(axis="y", labelcolor='#b93819')  # ZEROSET_COLOR)

    #major_x_locator = YearLocator(4)
    #ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    #minor_x_locator = YearLocator(1)
    #ax1.xaxis.set_minor_locator(minor_x_locator)

    #major_y_locator = MultipleLocator(5000)
    #ax1.yaxis.set_major_locator(major_y_locator)

    #minor_y_locator = MultipleLocator(2500)
    #ax1.yaxis.set_minor_locator(minor_y_locator)

    ax2.minorticks_on()
    #ax2.yaxis.set_minor_locator(MultipleLocator(2000))
    #ax2.yaxis.set_minor_locator(MultipleLocator(1000))

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.set_ylim([-1000, 29000])
    ax2.set_ylim([25500, 36900])
    ax1.legend(loc='upper left', fontsize=14)
    ax2.legend(loc='upper right',  bbox_to_anchor=(0.9, 1), fontsize=14)
    ax1.grid()
    title = "EDAC counter from MEX from Jan 1st, 2004 to Jul 30th, 2024"
    fig.suptitle(title,
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_rates_only():
    """
    MEX EDAC daily rate only
    """
    df = read_resampled_df()
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
    fig.suptitle("MEX EDAC daily rate between Jan 2004 and Apr 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_rolling_rate():
    df = read_rolling_rates(5)
    rate_mean = round(df['daily_rate'].mean(), 3)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["daily_rate"],
             label=f'MEX EDAC daily rate, mean = {rate_mean} counts per day',
             color=BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)

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
    fig.suptitle("MEX EDAC daily rate between Jan 2004 and Apr 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


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
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    # sunspots_smoothed = savgol_filter(df_sun["daily_sunspotnumber"],
    # SUNSPOTS_SAVGOL, 3)

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()
    ax2.plot(
        detrended_df["date"],
        detrended_df["gcr_component"],
        label="Savitzky-Golay fit",
        color=RAW_EDAC_COLOR
    )

    ax1.plot(
        df_sun["date"],
        df_sun["daily_sunspotnumber"],
        label="Sunspot number",
        color=SSN_COLOR
    )

    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Sunspot number", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
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
    plt.suptitle("MEX EDAC GCR component with sunspot number",
                 fontsize=FONTSIZE_TITLE)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.subplots_adjust(hspace=0.5)
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


def process_sidc_ssn():
    """
    Processes the SSN data from SIDC
    Parameters
    ----------
    file_path : Path
        The location of the file containing the SSN data

    Returns
    -------
    pd.DataFrame
        Dataframe containing the sunspot number
        for each date
    """
    column_names = [
        "year",
        "month",
        "day",
        "date_fraction",
        "daily_sunspotnumber",
        "std",
        "observations",
        "status",
    ]
    df_sun = pd.read_csv(RAW_DATA_DIR / "SN_d_tot_V2.0.csv",
                         names=column_names, sep=";")
    df_sun = df_sun[df_sun["daily_sunspotnumber"] >= 0]

    df_sun["date"] = pd.to_datetime(df_sun[["year", "month", "day"]])
    df_sun = df_sun[["date", "daily_sunspotnumber"]]
    return df_sun


def plot_rates_all():
    # Figure for thesis
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
    print(detrended_df)
    # print(detrended_df.sort_values(by="detrended_rate"))
    first_date = detrended_df['date'].iloc[0]
    last_date = detrended_df['date'].iloc[-1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 10))

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

    # minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(100))
    ax3.yaxis.set_minor_locator(MultipleLocator(25))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper left', bbox_to_anchor=(0.05, 1))
    plt.subplots_adjust(hspace=0.1)
    fig.suptitle("MEX EDAC count rates with the solar activity cycle",
                 fontsize=FONTSIZE_TITLE, y=0.92)
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
    )  # Save selected raw EDAC to file
    df.to_csv(
        LOCAL_DIR / "events" /
        f"EDACrate_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )  # Save selected EDAC rate to file
    edac_change.to_csv(
        LOCAL_DIR / "events" /
        f"EDACrate_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )
    print(df)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(10, 7.5))
    ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="MEX EDAC", s=3, color=RAW_EDAC_COLOR)
    #ax1.scatter(df["date"], df["edac_first"],
    #            label="MEX EDAC", color=RAW_EDAC_COLOR)

    ax2.plot(df["date"], df["daily_rate"], marker="o",
             label="EDAC count rate", color=RATE_EDAC_COLOR)

    ax3.plot(df["date"], df["detrended_rate"], marker="o",
             label="Detrended count rate", color=DETRENDED_EDAC_COLOR)
    # ax3.plot(
    # df["date"],
    # df["standardized_rate"],
    # marker="o",
    # label="Standardized EDAC count rate",
    # color=STANDARDIZED_EDAC_COLOR
    # )
    ax3.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("EDAC count", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC count rate", fontsize=FONTSIZE_AXES_LABELS)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)
    ax3.set_ylabel("Detrended count rate", fontsize=FONTSIZE_AXES_LABELS)
    ax3.tick_params(axis="x", rotation=10)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax3.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(5))

    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    """
    # ax1.axvline(x=datetime.strptime("2013-12-23", "%Y-%m-%d"),
                linestyle='dashed',
                color='black',
                label='Start of Forbush Decrease')
    """
    #ax2.set_ylim([-1, 6])
    #ax3.set_ylim([-2, 6])
    ax3.set_xlim(startdate, enddate)
    ax2.set_xlim(startdate, enddate)
    ax1.set_xlim(startdate, enddate)
    major_ticks = pd.date_range(start=startdate, end=enddate, freq='7D')
    ax1.set_xticks(major_ticks)

    """
    major_ticks_locations = [
        pd.to_datetime('2024-05-20 12:00:00')
        + pd.Timedelta(days=7 * i)
        for i in range(-2, 3)]
    ax1.set_xticks(major_ticks_locations)
    """
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    # ax3.legend(fontsize=FONTSIZE_LEGENDS,
    # loc='upper right', bbox_to_anchor=(0.9, 1))
    fig.suptitle("",
                 fontsize=16)
    # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    fig.subplots_adjust(top=0.94)
    plt.savefig(LOCAL_DIR / 'events' /
                f'edac_{startdate_string}{enddate_string}.png',
                dpi=300, transparent=False)
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
        fig.suptitle(f'{str(date.date())}')
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


def plot_histogram_rates():
    # Figure for thesis
    """
    Plot histogram distribution
    of detrended EDAC count rate"""
    df = read_detrended_rates()
    data = df["detrended_rate"]
    print(df.sort_values(by='detrended_rate'))
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
    bin_edges = np.arange(int(min_rate)-1, int(max_rate)+1, 0.5)
    print("bin_Edges: ", bin_edges)
    counts, bin_edges = np.histogram(data, bins=bin_edges,
                                     density=False)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 7))
    plt.hist(data, bins=bin_edges, color=DETRENDED_EDAC_COLOR,
             edgecolor="black")
    plt.xticks(np.arange(int(min_rate)-1,
                         int(max_rate)+1, 2))
    plt.gca().xaxis.set_minor_locator(
        plt.MultipleLocator(1))

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

    plt.title("Detrended rate distribution",
              fontsize=FONTSIZE_TITLE)
    plt.xlabel("Detrended count rate [#/day]",
               fontsize=FONTSIZE_AXES_LABELS)
    plt.ylabel("Occurrences", fontsize=FONTSIZE_AXES_LABELS)
    plt.legend(fontsize=12)
    plt.xlim(-2.5, 8)
    plt.grid()
    plt.show()


def create_stormy_plots():
    """
    Create plots of all stormy days, including all
    days which are a part of a Forbush Decrease
    and all dates with rates
    above the UPPER_THRESHOLD
    """
    print("Creating plots of all SWEET stormy dates")
    df_dates = read_stormy_sweet_dates()
    date_list = df_dates["date"].tolist()
    event_type_list = df_dates["type"].tolist()
    folder_name = "stormy_sweet"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_event_plots():
    """
    Creates plots of all SWEET events, including
    SEP events and Forbush decreases
    """
    print("------  Creating plots of SWEET events -------")
    df = read_sweet_event_dates()
    date_list = df["date"].tolist()
    event_type_list = df["type"].tolist()
    folder_name = "sweet_events"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_fd_plots():
    print("----- Creating plots of SWEET Forbush decreases ------")
    df_dates = read_sweet_forbush_decreases()
    date_list = df_dates["date"].tolist()
    event_type_list = df_dates["type"].tolist()
    folder_name = "forbush_decreases_sweet"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_sep_plots():
    print("---- Creating plots of SWEET SEP events -------")
    df_dates = read_sweet_sep_events()
    df_dates = df_dates.sort_values(by='date')
    date_list = df_dates["date"].tolist()
    event_type_list = df_dates["type"].tolist()
    folder_name = "sep_sweet"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_msl_rad_dates_sep_plots():
    print("---- Creating plots of MSL RAD SEP events -------")
    df_dates = read_sep_events_rad()
    date_list = df_dates["onset_time"].tolist()
    event_type_list = ['SEP']*len(date_list)
    folder_name = "msl_rad_dates_sep"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_maven_dates_sep_plots():
    print("---- Creating plots of MAVEN SEP events -------")
    df_dates = read_sep_events_maven()
    date_list = df_dates["onset_time"].tolist()
    event_type_list = ['SEP']*len(date_list)
    folder_name = "maven_dates_sep"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_msl_rad_dates_fd_plots():
    print("---- Creating plots of MSL RAD Forbush decreases -------")
    df_dates = read_forbush_decreases_rad()
    date_list = df_dates["onset_time"].tolist()
    event_type_list = ['FD']*len(date_list)
    folder_name = "msl_rad_dates_fd"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_msl_rad_fd_dates_sweet_found():
    print("---- Creating plots of MSL RAD Forbush decreases \
          that SWEET found -------")
    df_dates = read_msl_rad_fd_validation_result()
    df_dates = df_dates[df_dates['Fd_found']]
    # df_dates = df_dates[df_dates['Fd_found'] == True]
    date_list = df_dates["onset_time"].tolist()
    event_type_list = ['FD']*len(date_list)
    folder_name = "msl_rad_fd_found"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_maven_dates_fd_plots():
    print("---- Creating plots of SWEET events found in literature -------")
    df_dates = read_forbush_decreases_maven()
    date_list = df_dates["onset_time"].tolist()
    event_type_list = ['SEP']*len(date_list)
    folder_name = "maven_dates_sep"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)
    print("---- Creating plots of MSL RAD Forbush decreases -------")


def create_matched_plots():
    """
    Create plots of all SWEET event dates
    that have found a match in literature
    """
    print("skrr")


def plot_stormy_detection():
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df = read_detrended_rates()
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], SUNSPOTS_SAVGOL, 3
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))

    ax1.plot(df["date"], df["daily_rate"], label="EDAC count rate",
             COLOR=RATE_EDAC_COLOR)
    ax1.plot(df["date"], df["gcr_component"], label="Savitzky-Golay fit",
             color=RATE_FIT_COLOR)

    ax2.plot(
        df["date"],
        df["standardized_rate"],
        label="Standardized count rate",
        color=STANDARDIZED_EDAC_COLOR
    )
    ax2.axhline(
        UPPER_THRESHOLD,
        label="Threshold: " + str(UPPER_THRESHOLD),
        color=THRESHOLD_COLOR
    )
    # ax2.axhline(lower_threshold, color= thresholdcolor,
    #  label='Threshold: ' + str(lower_threshold))
    ax3.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Count rate [#/day]", fontsize=10)
    ax2.set_ylabel("Standardized count rate [#/day]", fontsize=10)
    ax3.plot(
        df_sun["date"],
        df_sun["daily_sunspotnumber"],
        color=SSN_COLOR,
        label="Number of sunspots",
    )
    ax3.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=SSN_SMOOTHED_COLOR,
        label="Smoothed sunspots",
    )
    ax3.set_ylabel("Sunspot number")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    # plt.show()


def plot_stormy_days_bin():
    """
    Plot the number of stormy days
    found in six month bins
    together with the sunspot number
    """
    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)

    spike_df = read_stormy_sweet_dates()
    sep_df = spike_df[spike_df["type"] == "SEP"]
    forbush_df = spike_df[spike_df["type"] == "Forbush"]
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], savgolwindow_sunspots, 3
    )

    spike_df["6_month_group"] = spike_df["date"].apply(group_by_6_months)
    sep_df["6_month_group"] = sep_df["date"].apply(group_by_6_months)
    forbush_df["6_month_group"] = forbush_df["date"].apply(group_by_6_months)

    grouped_df = spike_df.groupby("6_month_group").size().reset_index()
    grouped_sep = sep_df.groupby("6_month_group").size().reset_index()
    grouped_fd = forbush_df.groupby("6_month_group").size().reset_index()

    grouped_df.columns = ["datebin", "counts"]
    grouped_sep.columns = ["datebin", "counts"]
    grouped_fd.columns = ["datebin", "counts"]
    grouped_df["datebin"] = grouped_df["datebin"] + pd.DateOffset(months=3)
    grouped_sep["datebin"] = grouped_sep["datebin"] + pd.DateOffset(months=3)
    grouped_fd["datebin"] = grouped_fd["datebin"] + pd.DateOffset(months=3)
    stormy_total = grouped_df["counts"].sum()
    # sep_total = grouped_sep['counts'].sum()
    # fd_total = grouped_fd['counts'].sum()

    print("Number of stormy days: ", stormy_total)
    # spike_df.to_csv(path + 'stormydays.txt', sep='\t', index=False)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    spikecolor = "#377eb8"
    suncolor = "#a65628"
    sepcolor = "#ff7f00"
    ax1.plot(
        grouped_df["datebin"],
        grouped_df["counts"],
        marker="o",
        color=spikecolor,
        label="Total number of events",
    )
    ax2 = ax1.twinx()
    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Number of stormy days", fontsize=10)
    ax2.set_ylabel("Sunspot number")
    ax1.tick_params(axis="y", labelcolor=spikecolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Sunspot number and number of stormy days in 6 month bins")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    suncolor = "#a65628"
    sepcolor = "#ff7f00"
    ax1.plot(
        grouped_sep["datebin"],
        grouped_sep["counts"],
        marker="o",
        color=sepcolor,
        label="SEP",
        linewidth=1,
        alpha=1,
    )
    ax2 = ax1.twinx()
    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Number of SEP events", fontsize=10)
    ax2.set_ylabel("Sunspot number")
    ax1.tick_params(axis="y", labelcolor=spikecolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Sunspot number and number of SEPs in 6 month bins")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    suncolor = "#a65628"
    fdcolor = "#984ea3"
    ax1.plot(
        grouped_fd["datebin"],
        grouped_fd["counts"],
        marker="o",
        color=fdcolor,
        label="FD",
        linewidth=1,
        alpha=1,
    )
    ax2 = ax1.twinx()

    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Number of Forbush decreases", fontsize=10)
    ax2.set_ylabel("Sunspot number")
    ax1.tick_params(axis="y", labelcolor=spikecolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Sunspot number and number of FDs in 6 month bins")
    plt.show()


def plot_sweet_events_binned():
    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)
    stormy_days_df = read_stormy_sweet_dates()
    print(f"Number of SWEET stormy days is {len(stormy_days_df)}")
    event_df = read_sweet_event_dates()
    sep_df = event_df[event_df["type"] == "SEP"]
    # sep_df = read_sweet_sep_events()
    # sep_df["date"] = sep_df["start_date"]
    print(f'Number of SWEET SEP events: {len(sep_df)}')
    # forbush_df = read_sweet_zero_days()
    forbush_df = event_df[event_df["type"] == "Fd"]
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
    ax1.plot(
        grouped_df["datebin"],
        grouped_df["counts"],
        marker="o",
        color=all_events_color,
        label="Total number of events",
    )
    ax2 = ax1.twinx()
    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.xaxis.set_major_locator(YearLocator(4))
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)

    ax1.set_ylim([0, grouped_df["counts"].max()+2])
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Number of SWEET events", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Sunspot number", fontsize=FONTSIZE_AXES_LABELS)
    ax1.tick_params(axis="y", labelcolor=all_events_color)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(2))

    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax1.legend(loc="upper left", fontsize=FONTSIZE_LEGENDS)
    ax2.legend(loc="upper right", fontsize=FONTSIZE_LEGENDS,
               bbox_to_anchor=(0.9, 1))

    plt.title("Sunspot number and number of SWEET events in 6 month bins",
              fontsize=FONTSIZE_TITLE,
              pad=10)
    plt.show()
    plt.close()
###########
    fig, ax1 = plt.subplots(figsize=(10, 7.5))
    ax1.plot(
        grouped_sep["datebin"][:-1],
        grouped_sep["counts"][:-1],
        marker="o",
        color=sepcolor,
        label="Number of SWEET SEP events",
        linewidth=1,
        alpha=1,
    )
    ax2 = ax1.twinx()
    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.xaxis.set_major_locator(YearLocator(4))
    ax1.minorticks_on()
    ax1.tick_params(which='minor', length=6)
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(20))

    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)
    ax1.set_ylim([0, grouped_sep["counts"].max()+2])
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Number of SWEET SEP events", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Sunspot number", fontsize=FONTSIZE_AXES_LABELS)
    ax1.tick_params(axis="y", labelcolor=sepcolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc="upper left", fontsize=FONTSIZE_LEGENDS)
    ax2.legend(loc="upper right", fontsize=FONTSIZE_LEGENDS,
               bbox_to_anchor=(0.9, 1))
    plt.title("Sunspot number and number of SWEET SEP events in 6 month bins",
              fontsize=FONTSIZE_TITLE,
              pad=10)
    plt.show()
####
    fig, ax1 = plt.subplots(figsize=(10, 7.5))

    ax1.plot(
        grouped_fd["datebin"],
        grouped_fd["counts"],
        marker="o",
        color=fdcolor,
        label="Number of Forbush decreases",
        linewidth=1,
        alpha=1,
    )
    ax2 = ax1.twinx()

    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.xaxis.set_major_locator(YearLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(20))

    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(2))

    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Number of Forbush decreases",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Sunspot number", fontsize=FONTSIZE_AXES_LABELS)
    ax1.tick_params(axis="y", labelcolor=fdcolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc="upper left", fontsize=FONTSIZE_LEGENDS)
    ax2.legend(loc="upper right", fontsize=FONTSIZE_LEGENDS,
               bbox_to_anchor=(0.9, 1))
    plt.title("Sunspot number and number of Forbush decreases in 6 month bins",
              fontsize=FONTSIZE_TITLE,
              pad=10)
    plt.show()


def group_extra_seps():
    # df = combine_lenient_sep_with_seplist()
    df = read_extra_sweet_sep_events()

    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)
    df["6_month_group"] = df["date"].apply(group_by_6_months)

    grouped_df = df.groupby("6_month_group").size().reset_index()
    grouped_df.columns = ["datebin", "counts"]
    grouped_df["datebin"] = grouped_df["datebin"] \
        + pd.DateOffset(months=3)
    stormy_total = grouped_df["counts"].sum()
    print("Number of SEPs: ", stormy_total)

    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], savgolwindow_sunspots, 3
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    spikecolor = "#377eb8"
    suncolor = "#a65628"
    ax1.plot(
        grouped_df["datebin"],
        grouped_df["counts"],
        marker="o",
        color=spikecolor,
        label="Total number of events",
    )
    # ax1.plot(grouped_sep['datebin'],grouped_sep['counts'],
    # marker='o', color=sepcolor,label='SEP',linewidth=0.5,alpha=0.5)
    # ax1.plot(grouped_fd['datebin'],grouped_fd['counts'],
    # marker='o', color=fdcolor, label='FD',linewidth=0.5, alpha=0.5)
    ax2 = ax1.twinx()
    # ax2.plot(df_sun['date'], df_sun['daily_sunspotnumber'],
    # label="Number of sunspots")
    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Number of stormy days", fontsize=10)
    ax2.set_ylabel("Sunspot number")
    ax1.tick_params(axis="y", labelcolor=spikecolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Number of SEPs and SSN in 6 month bins")
    plt.show()


def group_fds(file_path):
    """
    Group how many Forbush decreases
    there has been in
    6 month bins, and plot
    with the solar cycle
    """
    df = read_sweet_forbush_decreases(file_path)

    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)

    df["6_month_group"] = df["date"].apply(group_by_6_months)
    grouped_df = df.groupby("6_month_group").size().reset_index()
    grouped_df.columns = ["datebin", "counts"]
    grouped_df["datebin"] = grouped_df["datebin"] \
        + pd.DateOffset(months=3)
    stormy_total = grouped_df["counts"].sum()
    print("Number of forbush decreases: ", stormy_total)

    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], savgolwindow_sunspots, 3
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    spikecolor = "#377eb8"
    suncolor = "#a65628"
    ax1.plot(
        grouped_df["datebin"],
        grouped_df["counts"],
        marker="o",
        color=spikecolor,
        label="Total number of events",
    )
    # ax1.plot(grouped_sep['datebin'],grouped_sep['counts'],
    # marker='o', color=sepcolor,label='SEP',linewidth=0.5,alpha=0.5)
    # ax1.plot(grouped_fd['datebin'],grouped_fd['counts'],
    # marker='o', color=fdcolor, label='FD',linewidth=0.5, alpha=0.5)
    ax2 = ax1.twinx()
    # ax2.plot(df_sun['date'], df_sun['daily_sunspotnumber'],
    # label="Number of sunspots")
    ax2.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color=suncolor,
        label="Smoothed sunspots",
    )
    ax2.set_ylim([0, max(sunspots_smoothed + 10)])
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Number of stormy days", fontsize=10)
    ax2.set_ylabel("Sunspot number")
    ax1.tick_params(axis="y", labelcolor=spikecolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Number of Forbush decreases and SSN in 6 month bins")
    plt.show()


def plot_real_eruption_dates():
    """
    Plot the raw EDAC, EDAC count rate and
    the de-trended rate for the CME eruption dates"""
    raw_edac = read_rawedac(
    )
    standardized_df = read_detrended_rates()
    validation_df = read_cme_validation_results()
    folder_name = 'validation_result'
    if not os.path.exists(LOCAL_DIR / folder_name):
        os.makedirs(LOCAL_DIR / folder_name)

    for i in range(0, len(validation_df)):
        current_date = validation_df.iloc[i]["eruption_date"]
        event_status = validation_df.iloc[i]["result"]
        date_string = str(current_date.date()).replace(" ", "_")
        startdate = current_date - pd.Timedelta(days=21)
        enddate = current_date + pd.Timedelta(days=21)
        temp_raw = raw_edac.copy()
        temp_raw = temp_raw[
            (temp_raw["datetime"] > startdate) &
            (temp_raw["datetime"] < enddate)
        ]
        temp_standardized = standardized_df.copy()
        temp_standardized = temp_standardized[
            (temp_standardized["date"] > startdate) &
            (temp_standardized["date"] < enddate)
        ]
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 7))
        ax1.scatter(temp_raw["datetime"], temp_raw["edac"],
                    label="Raw EDAC", s=3,
                    color=RAW_EDAC_COLOR)
        ax2.plot(
            temp_standardized["date"],
            temp_standardized["daily_rate"],
            marker="o",
            label="EDAC count rate",
            color=RATE_EDAC_COLOR
        )
        ax3.plot(
            temp_standardized["date"],
            temp_standardized["detrended_rate"],
            marker="o",
            label="De-trended rate",
            color=DETRENDED_EDAC_COLOR
        )
        ax3.plot(
            temp_standardized["date"],
            temp_standardized["standardized_rate"],
            marker="o",
            label="Standardized EDAC count rate",
            color=STANDARDIZED_EDAC_COLOR
        )
        ax3.axvline(x=current_date, color="black", linestyle='dashed',
                    linewidth="1", label=current_date)
        ax3.axhline(UPPER_THRESHOLD, color=THRESHOLD_COLOR)
        ax3.set_xlabel("Date", fontsize=12),
        ax1.set_ylabel("EDAC count", fontsize=12)
        ax2.set_ylabel("EDAC count rate", fontsize=12)
        ax3.set_ylabel("De-trended count rate", fontsize=12)
        # Adjust the rotation angle as needed
        ax3.tick_params(axis="x", rotation=20)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.legend()
        ax2.legend()
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
        fig.suptitle(f"{current_date.date()}, SWEET found: {event_status}")
        plt.tight_layout(pad=2.0)
        plt.savefig(
            LOCAL_DIR / folder_name / f"{date_string}",
            dpi=300,
            transparent=False,
        )

        # plt.show()
        plt.close()


def plot_compare_sweets_validations():
    """
    Create plots of EDAC count,
    EDAC count rate
    and detrended EDAC count rate
    for each CME eruption date in the database
    with the results from new SWEET
    and old SWEET

    """
    validation_old = read_cme_validation_results_old()
    validation_new = read_cme_validation_results()
    folder_name = "bothsweets_validation"

    raw_edac = read_rawedac(
    )
    standardized_df = read_detrended_rates()

    if not os.path.exists(LOCAL_DIR / folder_name):
        os.makedirs(LOCAL_DIR / folder_name)

    for i in range(0, len(validation_new)):
        current_date = validation_new.iloc[i]["eruption_date"]
        new_event_status = validation_new.iloc[i]["result"]
        old_event_status = validation_old.iloc[i]["result"]
        date_string = str(current_date.date()).replace(" ", "_")
        startdate = current_date - pd.Timedelta(days=21)
        enddate = current_date + pd.Timedelta(days=21)
        temp_raw = raw_edac.copy()
        temp_raw = temp_raw[
            (temp_raw["datetime"] > startdate) &
            (temp_raw["datetime"] < enddate)
        ]
        temp_standardized = standardized_df.copy()
        temp_standardized = temp_standardized[
            (temp_standardized["date"] > startdate) &
            (temp_standardized["date"] < enddate)
        ]
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 7))
        ax1.scatter(temp_raw["datetime"], temp_raw["edac"],
                    label="Raw EDAC", s=3,
                    color=RAW_EDAC_COLOR)
        ax2.plot(
            temp_standardized["date"],
            temp_standardized["daily_rate"],
            marker="o",
            label="EDAC count rate",
            color=RATE_EDAC_COLOR
        )
        ax3.plot(
            temp_standardized["date"],
            temp_standardized["detrended_rate"],
            marker="o",
            label="De-trended rate",
            color=DETRENDED_EDAC_COLOR
        )
        ax3.plot(
            temp_standardized["date"],
            temp_standardized["standardized_rate"],
            marker="o",
            label="Standardized EDAC count rate",
            color=STANDARDIZED_EDAC_COLOR
        )
        ax3.axvline(x=current_date, color="black", linestyle='dashed',
                    linewidth="1", label=current_date)
        ax3.axhline(UPPER_THRESHOLD, color=THRESHOLD_COLOR)
        ax3.set_xlabel("Date", fontsize=12),
        ax1.set_ylabel("EDAC count", fontsize=12)
        ax2.set_ylabel("EDAC count rate", fontsize=12)
        ax3.set_ylabel("De-trended count rate", fontsize=12)
        # Adjust the rotation angle as needed
        ax3.tick_params(axis="x", rotation=20)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.legend()
        ax2.legend()
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
        fig.suptitle(f"{current_date.date()}, new found: {new_event_status}\
                     old found: {old_event_status}")
        plt.tight_layout(pad=2.0)
        plt.savefig(
            LOCAL_DIR / folder_name / f"{date_string}",
            dpi=300,
            transparent=False,
        )

        # plt.show()
        plt.close()


def plot_real_sep_onsets():
    """
    Plot the raw EDAC, EDAC count rate and
    the de-trended rate for the SEP onset times
    """
    raw_edac = read_rawedac(
    )
    standardized_df = read_detrended_rates()
    validation_df = read_sep_validation_results()
    folder_name = 'validation_result'
    if not os.path.exists(SEP_VALIDATION_DIR / folder_name):
        os.makedirs(SEP_VALIDATION_DIR / folder_name)

    for i in range(0, len(validation_df)):
        current_date = validation_df.iloc[i]["onset_time"]
        event_status = validation_df.iloc[i]["SEP_found"]
        date_string = str(current_date.date()).replace(" ", "_")
        startdate = current_date - pd.Timedelta(days=21)
        enddate = current_date + pd.Timedelta(days=21)
        temp_raw = raw_edac.copy()
        temp_raw = temp_raw[
            (temp_raw["datetime"] > startdate) &
            (temp_raw["datetime"] < enddate)
        ]
        temp_standardized = standardized_df.copy()
        temp_standardized = temp_standardized[
            (temp_standardized["date"] > startdate) &
            (temp_standardized["date"] < enddate)
        ]
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 7))
        ax1.scatter(temp_raw["datetime"], temp_raw["edac"],
                    label="Raw EDAC", s=3,
                    color=RAW_EDAC_COLOR)
        ax2.plot(
            temp_standardized["date"],
            temp_standardized["daily_rate"],
            marker="o",
            label="EDAC count rate",
            color=RATE_EDAC_COLOR
        )
        ax3.plot(
            temp_standardized["date"],
            temp_standardized["detrended_rate"],
            marker="o",
            label="De-trended rate",
            color=DETRENDED_EDAC_COLOR
        )
        ax3.plot(
            temp_standardized["date"],
            temp_standardized["standardized_rate"],
            marker="o",
            label="Standardized EDAC count rate",
            color=STANDARDIZED_EDAC_COLOR
        )
        ax3.axvline(x=current_date, color="black", linestyle='dashed',
                    linewidth="1", label=current_date)
        ax3.axhline(UPPER_THRESHOLD, color=THRESHOLD_COLOR)
        ax3.set_xlabel("Date", fontsize=12),
        ax1.set_ylabel("EDAC count", fontsize=12)
        ax2.set_ylabel("EDAC count rate", fontsize=12)
        ax3.set_ylabel("De-trended count rate", fontsize=12)
        # Adjust the rotation angle as needed
        ax3.tick_params(axis="x", rotation=20)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.legend()
        ax2.legend()
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
        fig.suptitle(f"{current_date.date()}, SWEET found: {event_status}")
        plt.tight_layout(pad=2.0)
        plt.savefig(
            SEP_VALIDATION_DIR / folder_name / f"{date_string}",
            dpi=300,
            transparent=False,
        )

        # plt.show()
        plt.close()


def plot_solar_cycle():
    df = process_sidc_ssn()
    sunspots_smoothed = savgol_filter(
        df["daily_sunspotnumber"], SUNSPOTS_SAVGOL-200, 3
    )
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["daily_sunspotnumber"],

             label='Daily sunspot number',
             color=SSN_COLOR,  # BRAT_GREEN,
             linewidth=2)

    ax1.plot(df["date"], sunspots_smoothed,
             color=SSN_SMOOTHED_COLOR,
             label="Smoothed sunspot number")
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Sunspot number [#]",
                   fontsize=FONTSIZE_AXES_LABELS)

    major_x_locator = YearLocator(20)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(10)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    major_y_locator = MultipleLocator(50)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(25)
    ax1.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()
    # fig.suptitle("brat")
    fig.suptitle("The sunspot number from January 1810 to March 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_detrended_rates():
    df = read_detrended_rates()

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["detrended_rate"],
             label='Detrended count rate',
             color=DETRENDED_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Detrended count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)

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


def plot_detrended_rates_with_solar_cycle():
    df = read_detrended_rates()

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["detrended_rate"],
             label='Detrended count rate',
             color=DETRENDED_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Detrended count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS,
                   color=DETRENDED_EDAC_COLOR)
    ax2 = ax1.twinx()
    print("df: ", df)
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-07-31", "%Y-%m-%d")
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    index_end = np.where(df_sun["date"] == end_date)[0][0]
    df_sun = df_sun.iloc[index_exact:index_end]
    print("df_sun: ", df_sun)
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], SUNSPOTS_SAVGOL, 3
    )
    ax2.plot(df_sun["date"], df_sun["daily_sunspotnumber"],
             label='Daily sunspot number',
             color=SSN_COLOR,  # BRAT_GREEN,
             linewidth=2,
             alpha=0.5)

    ax2.plot(df["date"], sunspots_smoothed,
             color=SSN_SMOOTHED_COLOR,
             label="Smoothed sunspot number",
             alpha=0.5)
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Sunspot number [#]",
                   fontsize=FONTSIZE_AXES_LABELS,
                   color=SSN_COLOR)
    ax2.set_ylim([-120, max(sunspots_smoothed + 100)])
    major_x_locator = YearLocator(20)
    ax2.xaxis.set_major_locator(major_x_locator)
    ax2.minorticks_on()
    minor_x_locator = YearLocator(10)
    ax2.xaxis.set_minor_locator(minor_x_locator)
    ax1.grid()
    major_y_locator = MultipleLocator(50)
    ax2.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(25)
    ax2.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc="upper right", bbox_to_anchor=(0.9, 1))
    ax1.grid()

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
    fig.suptitle("MEX EDAC detrended count rate with the solar cycle",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def show_timerange_counter_countrate(startdate, enddate):
    """
    For a time period between startdate and enddate,
    create plot of the raw EDAC, the count rate,
    de-trended count rate, standardized count rate
    """
    raw_edac = read_rawedac()
    filtered_raw = raw_edac.copy()
    filtered_raw = filtered_raw[
        (filtered_raw["datetime"] > startdate) &
        (filtered_raw["datetime"] < enddate)
    ]

    df = read_detrended_rates()
    df = df[(df["date"] > startdate) & (df["date"] < enddate)]
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

    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   figsize=(8, 5))
    # ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
    #           label="MEX EDAC", s=3, color=RAW_EDAC_COLOR)
    ax1.scatter(df["date"], df["edac_first"],
                label="MEX EDAC", color="midnightblue")

    ax2.plot(df["date"], df["daily_rate"], marker="o",
             label="EDAC daglig rate", color="maroon")
    """
    ax3.plot(
        df["date"],
        df["standardized_rate"],
        marker="o",
        label="Standardized EDAC count rate",
        color=STANDARDIZED_EDAC_COLOR
    )
    """
    ax2.set_xlabel("Dato", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("EDAC teller [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC tellerrate [#/dag]", fontsize=FONTSIZE_AXES_LABELS)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)
    ax2.tick_params(axis="x", rotation=10)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    # ax2.axvline(x=datetime.strptime("2014-09-02", "%Y-%m-%d"),
    # linestyle='dashed',
    # color='black')
    ax2.set_ylim([-1, 6])
    ax2.set_ylim([-1, 18])
    ax2.set_xlim(startdate, enddate)
    ax1.set_xlim(startdate, enddate)

    major_ticks_locations = [
        pd.to_datetime('2017-09-11 00:00:00')
        + pd.Timedelta(days=7 * i)
        for i in range(-3, 4)
    ]
    # One week away, up to five weeks
    ax1.set_xticks(major_ticks_locations)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax1.grid()
    ax2.grid()

    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    # ax3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right',
    # bbox_to_anchor=(0.9, 1))
    fig.suptitle("Romvær-hendelse i september 2017", fontsize=16)
    # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    fig.subplots_adjust(top=0.94)
    plt.savefig(LOCAL_DIR / 'events' /
                f'edac_{startdate_string}{enddate_string}.png',
                dpi=300, transparent=False)
    plt.show()


def plot_zero_set_and_detrended():
    detrended_df = read_detrended_rates()
    zeroset_df = read_zero_set_correct()
    # print(detrended_df.sort_values(by="detrended_rate"))
    first_date = detrended_df['date'].iloc[0]
    last_date = detrended_df['date'].iloc[-1]
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))


    ax2.plot(
        detrended_df["date"],
        detrended_df["detrended_rate"],
        label="MEX EDAC daglig rate",
        color="maroon"
    )
    ax1.plot(zeroset_df["datetime"], zeroset_df["edac"],
             label='MEX EDAC data',
             color="midnightblue",
             linewidth=2)
    


    ax2.set_xlabel("Dato", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC teller-rate [#/dag]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("EDAC teller [#]",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_xlim([first_date, last_date])
    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)

    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # minor_y_locator = MultipleLocator(1)
    # ax1.yaxis.set_major_locator(MultipleLocator(5))
    # ax1.yaxis.set_minor_locator(MultipleLocator(1))
    # ax2.yaxis.set_major_locator(MultipleLocator(5))
    # ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.grid()
    ax2.grid()

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    # ax3.legend(loc='upper left', bbox_to_anchor=(0.05, 1))
    plt.subplots_adjust(hspace=0.1)
    fig.suptitle("MEX EDAC data mellom januar 2004 og juli 2024",
                 fontsize=FONTSIZE_TITLE, y=0.94)
    plt.show()

# IMA 

def plot_mex_ima_bg_counts_time_interval(start_date, end_date):
    df = read_mex_ima_bg_counts()
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['datetime'], df['bg_counts'],
               label="Background counts",
               marker='o')
    # ax.scatter(df['datetime'], df['total_counts'], s=0.5,
    #           label="Total counts")

    ax.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    weeks_in_interval = (end_date-start_date).days//7
    print(weeks_in_interval)
    major_ticks_locations = [
        start_date
        + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    # One week away, up to five weeks
    ax.set_xticks(major_ticks_locations)
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=0)
    ax.set_yscale('log')
    ax.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax.set_ylabel("Counts",  fontsize=FONTSIZE_AXES_LABELS)
    ax.legend()
    ax.set_title("MEX/ASPERA-3 IMA bg. counts", fontsize=FONTSIZE_TITLE,
                 pad=2)
    ax.grid()
    plt.show()
    

def plot_ima_counts_and_sweet(start_date, end_date):
    """
    Plot EDAC counter with the IMA bg. counts,
    the count rate and the GCR component
    and the detrended count rate
    
    """
    # df_ima = read_mex_ima_bg_counts()
    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_date) & (df_raw["datetime"] <= end_date)]
    print(df_raw)

    df_ima = clean_up_mex_ima_bg_counts()
    df_ima = df_ima[(df_ima["datetime"] >= start_date) & (df_ima["datetime"] <= end_date)]
    df_sweet = read_detrended_rates()
    df_sweet = df_sweet[(df_sweet["date"] > start_date) & (df_sweet["date"] < end_date)]
    print(df_sweet)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                   figsize=(8, 6))
    
    ax1.plot(df_ima["datetime"], df_ima["bg_counts"],
             label="IMA background counts",
             color=IMA_COLOR)
    
    ax4 = ax1.twinx()
    ax4.scatter(df_raw["datetime"], df_raw["edac"],
                label='Raw MEX EDAC',
                color=RAW_EDAC_COLOR,
                marker='o',
                s=5)
    
    #ax0.scatter(df_raw["datetime"], df_raw["edac"],
    #            label='Raw MEX EDAC',
    #            color=RAW_EDAC_COLOR,
    #            linewidth=2,)
    ax2.plot(df_sweet["date"], df_sweet["daily_rate"],
             marker='o',
             color=RATE_EDAC_COLOR,
             label='MEX EDAC count rate')
    
    ax2.plot(df_sweet["date"], df_sweet["gcr_component"],
             marker='o',
             color=RATE_FIT_COLOR,
             label='GCR component')
    
    ax3.plot(df_sweet["date"], df_sweet["detrended_rate"],
             marker='o',
             color=DETRENDED_EDAC_COLOR,
             label='Detrended EDAC count rate')
    
    ax3.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    
    
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Count rate", fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_ylabel("Detrended count rate", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("IMA bg. counts", fontsize=FONTSIZE_AXES_LABELS)
    ax4.set_ylabel("MEX EDAC count", fontsize=FONTSIZE_AXES_LABELS)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)

    fig.suptitle("MEX EDAC and IMA bg. counts for January 2012 event",
                fontsize=FONTSIZE_TITLE,
                 y=0.95)
    
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_ylim([-1, df_sweet["daily_rate"].max()+4])
    ax3.yaxis.set_major_locator(MultipleLocator(4))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    ax4.yaxis.set_major_locator(MultipleLocator(10))
    ax4.yaxis.set_minor_locator(MultipleLocator(5))

    max_y = df_sweet["detrended_rate"].max()
    ax2.set_ylim([-1, max_y+4])
    ax4.set_ylim([df_raw["edac"].min()-2, df_raw["edac"].max()+2])

    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
        + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    ax3.set_xticks(major_ticks_locations)
    ax3.xaxis.set_minor_locator(mdates.DayLocator())

    ax1.set_yscale('log')
    ax3.tick_params(axis="x", rotation=0)
    ax4.yaxis.tick_left()
    ax4.yaxis.set_label_position("left")

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax4.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax4.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax3.legend()
    ax2.legend(loc='upper left')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()
    handles = handles1 + handles4
    labels = labels1 + labels4
    ax1.legend(handles, labels, loc='upper left')
    plt.show()
    # print(df_ima)


    df_ima['time_difference'] = df_ima['datetime'].diff()
    df_ima['time_difference_in_minutes'] = \
        df_ima['time_difference'].dt.total_seconds() / 60
    

    df_ima.to_csv("test_ima.txt")


def plot_ima_counts_and_sweet_v2(start_date, end_date):
    """
    Plots IMA bg counts, the EDAC counter
    and the detrended count

    Thesis figure
    """

    # df_ima = read_mex_ima_bg_counts()
    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_date) & (df_raw["datetime"] <= end_date)]

    df_ima = clean_up_mex_ima_bg_counts()
    df_ima = df_ima[(df_ima["datetime"] >= start_date) & (df_ima["datetime"] <= end_date)]
    df_sweet = read_detrended_rates()
    # print(df_sweet)
    # print(df_sweet.sort_values(by="detrended_rate", ascending=False).iloc[9:20])
    df_sweet = df_sweet[(df_sweet["date"] > start_date) & (df_sweet["date"] < end_date)]
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                   figsize=(10, 6))
    
    ax1.plot(df_ima["datetime"], df_ima["bg_counts"],
             label="IMA background counts",
             color=IMA_COLOR)
    
    ax2.scatter(df_raw["datetime"], df_raw["edac"],
                label='MEX EDAC counter value',
                color=RAW_EDAC_COLOR,
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
    
    
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("IMA bg. counts", fontsize=FONTSIZE_AXES_LABELS)

    
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)

    # fig.suptitle("MEX EDAC and IMA bg. counts during January 2014 event",
    #            fontsize=FONTSIZE_TITLE,
    #             y=0.95)
    
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
        + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    ax3.set_xticks(major_ticks_locations)
    ax3.xaxis.set_minor_locator(mdates.DayLocator())

    ax1.set_yscale('log')
    ax3.tick_params(axis="x", rotation=0)

    # ax1.yaxis.tick_right()
    # ax1.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax3.set_ylim([-1, 8])

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax3.legend()
    ax2.legend(loc='upper left')
    ax1.legend(loc='upper left')

    if not os.path.exists(LOCAL_DIR / "events/sweet_ima_comparison"):
        os.makedirs(LOCAL_DIR / "events/sweet_ima_comparison")

    plt.savefig(LOCAL_DIR / 'events/sweet_ima_comparison' /
                f'sweet_ima_{str(start_date.date())}.png',
                dpi=300, transparent=False)
    
    # plt.close()
    plt.show()
    # print(df_ima)
    

    df_ima['time_difference'] = df_ima['datetime'].diff()
    df_ima['time_difference_in_minutes'] = \
        df_ima['time_difference'].dt.total_seconds() / 60
    
    print(df_ima['time_difference_in_minutes'].max())
    df_ima.to_csv("test_ima.txt")


def plot_ima_counts_all():
    # df_ima = read_mex_ima_bg_counts()
    df_ima = clean_up_mex_ima_bg_counts()
    # print(df_ima)
    fig, ax = plt.subplots(1, sharex=True,
                                   figsize=(8, 5))
    ax.plot(df_ima['datetime'], df_ima['bg_counts'])
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.xaxis.set_major_locator(YearLocator(4))
    ax.xaxis.set_minor_locator(YearLocator(1))
    ax.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax.set_ylabel("Counts",  fontsize=FONTSIZE_AXES_LABELS)
    ax.grid()
    # ax.plot(df_ima['datetime'], df_ima['total_counts'])
    plt.show()


def compare_sweet_and_ima_bg(eventdate):
    start_date = eventdate - pd.Timedelta(days=7)
    end_date = eventdate + pd.Timedelta(days=7)
    plot_ima_counts_and_sweet(start_date, end_date)

    start_date = eventdate - pd.Timedelta(days=10)
    end_date = eventdate + pd.Timedelta(days=10)
    # plot_raw_edac_scatter(start_date, end_date)


def plot_ima_with_solar_cycle():
    # df_ima = read_mex_ima_bg_counts()
    df_ima = clean_up_mex_ima_bg_counts()

    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-07-31", "%Y-%m-%d")
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    index_end = np.where(df_sun["date"] == end_date)[0][0]
    df_sun = df_sun.iloc[index_exact:index_end]

    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], SUNSPOTS_SAVGOL, 3
    )

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df_ima["datetime"], df_ima["bg_counts"],
             label='MEX IMA background counts',
             color=DETRENDED_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Counts",
                   fontsize=FONTSIZE_AXES_LABELS,
                   color=DETRENDED_EDAC_COLOR)
    ax2 = ax1.twinx()

    ax2.plot(df_sun["date"], df_sun["daily_sunspotnumber"],
            label='Daily sunspot number',
             color=SSN_COLOR,  # BRAT_GREEN,
             linewidth=2,
             alpha=1)
    
    ax2.plot(df_sun["date"], sunspots_smoothed,
            label='Daily smoothed sunspot number',
            color=BRAT_GREEN,
            linewidth=2,
            alpha=1)

    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Sunspot number [#]",
                   fontsize=FONTSIZE_AXES_LABELS,
                   color=SSN_COLOR)
    # ax2.set_ylim([-120, max(sunspots_smoothed + 100)])
    major_x_locator = YearLocator(4)
    ax2.xaxis.set_major_locator(major_x_locator)
    ax2.minorticks_on()
    ax1.set_yscale('log')

    minor_x_locator = YearLocator(10)
    ax2.xaxis.set_minor_locator(minor_x_locator)
    ax1.grid()
    major_y_locator = MultipleLocator(50)
    ax2.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(25)
    ax2.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc="upper right", bbox_to_anchor=(0.9, 1))
    ax1.grid()

    plt.show()


def create_sweet_mex_safe_modes_plots():
    print("---- Creating plots of MEX Safe Modes -------")
    df_dates = read_mex_safe_modes()
    date_list = df_dates["occurrence_date"].tolist()
    event_type_list = ['SEP?']*len(date_list)
    folder_name = "mex_safe_modes"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def plot_aspera_sw_moments(start_date, end_date):
    df = read_aspera_sw_moments()
    # print(df)
    print("end_Date: ", end_date)
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    print(df)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 6), sharex=True)
    ax1.plot(df["datetime"], df["speed"],
             color=IMA_COLOR)
    ax2.plot(df["datetime"], df["density"],
             color=IMA_COLOR)

    ax1.set_ylabel("Speed [km/s]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel(r"Density [$\mathrm{cm}^{-3}$]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.minorticks_on()
    ax1.yaxis.set_major_locator(MultipleLocator(100))
    ax1.yaxis.set_minor_locator(MultipleLocator(50))
    
    ax2.yaxis.set_major_locator(MultipleLocator(3))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    # major_ticks = pd.date_range(start=start_date, end=end_date, freq='7D')
    # ax2.set_xticks(major_ticks)
    # minor_ticks = pd.date_range(start=start_date, end=end_date, freq='D')
    # ax2.set_xticks(minor_ticks, minor=True)
    currentdate = datetime.strptime("2012-01-27", "%Y-%m-%d")
    major_ticks_locations = [
        currentdate
        + pd.Timedelta(days=7 * i)
        for i in range(-1, 2)]
    ax2.set_xticks(major_ticks_locations)
    minor_ticks_locations = [
        currentdate
        + pd.Timedelta(days= i)
        for i in range(-7, 8)]
    ax2.set_xticks(minor_ticks_locations, minor=True)
    
    ax1.grid()
    ax2.grid()
    fig.suptitle("MEX/ASPERA-3 IMA solar wind moments during January 2012 event",
                 fontsize=FONTSIZE_TITLE)

    # ax2.xaxis.set_major_locator(DayLocator(3))
    # ax2.xaxis.set_minor_locator(DayLocator(1))
    
    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    plt.tight_layout(pad=1.0)
    plt.show()

def create_sweet_ima_plots_for_largest_sweet_events():
    df_sweet = read_detrended_rates()
    #print(df_sweet)
    filtered_df = df_sweet.sort_values(by="detrended_rate", ascending=False).iloc[0:20]
    date_list = filtered_df["date"].tolist()
    print(date_list)

    for date in date_list:
        print(date)
        start_date = date - pd.Timedelta(days=7)
        end_date = date + pd.Timedelta(days=7)

        plot_ima_counts_and_sweet_v2(start_date, end_date)

def plot_ima_sweet_samplewise(start_time, end_time):
    """
    To investigate timing
    """
    # df_ima = read_mex_ima_bg_counts()
    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_time) & (df_raw["datetime"] <= end_time)]

    df_ima = clean_up_mex_ima_bg_counts()
    df_ima = df_ima[(df_ima["datetime"] >= start_time) & (df_ima["datetime"] <= end_time)]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   figsize=(10, 6))
    
    ax1.plot(df_ima["datetime"], df_ima["bg_counts"],
             label="IMA background counts",
             marker='o',
             markersize=5,
             color=IMA_COLOR)
    
    ax2.scatter(df_raw["datetime"], df_raw["edac"],
                label='MEX EDAC counter value',
                color=RAW_EDAC_COLOR,
                marker='o',
                s=5)
    
    # ax1.axvline(x=datetime.strptime("2024-05-20 05:35:43", "%Y-%m-%d %H:%M:%S"),
    #            linestyle='dashed',
    #            color='black',
    #            label='Start of IMA bg. counts enhancement')
    #ax2.axvline(x=datetime.strptime("2024-05-20 05:35:43", "%Y-%m-%d %H:%M:%S"),
    #            linestyle='dashed',
    #            color='black',
    #            label='Start of IMA bg. counts enhancement')
    
    ax2.set_xlabel("Time", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("IMA bg. counts", fontsize=FONTSIZE_AXES_LABELS)


    fig.suptitle("MEX EDAC and IMA bg. counts on March 2012 event",
                fontsize=FONTSIZE_TITLE,
                 y=0.95)
    
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(2.5))

    ax1.set_yscale('log')
    # ax1.yaxis.tick_right()
    # ax1.yaxis.set_label_position("right")
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position("right")
    ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %#d %H'))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 12)))

    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
    # ax2.xaxis.set_minor_locator(mdates.HourLocator())

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(axis="x", rotation=0)
    # start_time = datetime(2024, 5, 20, 0, 0, 0)  
    # end_time = datetime(2024, 5, 20, 23, 59, 59) 
    # ax2.set_xlim(start_time, end_time)

    ax1.grid()
    ax2.grid()

    ax2.legend(loc='lower right')
    ax1.legend(loc='lower right')
    if not os.path.exists(LOCAL_DIR / "events/sweet_ima_comparison"):
        os.makedirs(LOCAL_DIR / "events/sweet_ima_comparison")

    plt.savefig(LOCAL_DIR / 'events/sweet_ima_comparison' /
                f'sweet_ima_timing{str(start_date.date())}.png',
                dpi=300, transparent=False)
    

    # plt.close()
    plt.show()
    # print(df_ima)

if __name__ == "__main__":
    # plot_ima_counts_all()  April 3rd, 2023
    currentdate = datetime.strptime("2012-03-09", "%Y-%m-%d")
    #plot_raw_edac_scatter(datetime.strptime("2011-06-03", "%Y-%m-%d"), 
    #                      datetime.strptime("2011-06-10", "%Y-%m-%d"))
    # start_date = datetime.strptime("2023-12-01", "%Y-%m-%d")
    # end_date = datetime.strptime("2024-01-05", "%Y-%m-%d")
    start_date = currentdate - pd.Timedelta(days=7)
    end_date = currentdate + pd.Timedelta(days=7)
    # create_sweet_ima_plots_for_largest_sweet_events()
    # plot_raw_edac_scatter(start_date, end_date)
    # show_timerange(start_date, end_date)
    # plot_raw_and_zerosetcorrected()
    #  plot_raw_and_zerosetcorrected()
    # plot_ima_counts_and_sweet(start_date, end_date)
    start_time =  datetime.strptime("2012-03-04 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_time =  datetime.strptime("2012-03-11 23:59:45", "%Y-%m-%d %H:%M:%S")

    # plot_ima_counts_and_sweet_v2(start_date, end_date)
    plot_ima_sweet_samplewise(start_time, end_time)
    
    

    # plot_aspera_sw_moments(start_date, end_date)
    # plot_mex_ima_bg_counts_time_interval(start_date, end_date)
    # startdate = currentdate - pd.Timedelta(days=14)
    # enddate = currentdate + pd.Timedelta(days=14)
    # show_timerange(start_date, end_date)
    # create_sweet_mex_safe_modes_plots()