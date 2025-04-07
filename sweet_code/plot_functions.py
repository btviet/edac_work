import os
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
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
    RAD_E_COLOR,
    RAD_B_COLOR,
    TOOLS_OUTPUT_DIR
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
    read_rad_onsets,
    read_sep_database_events,
    read_forbush_decreases_database,
)
from scipy.signal import savgol_filter
# from scipy.stats import percentileofscore
from validate_cme_events import read_cme_validation_results
from validate_forbush_decreases import read_msl_rad_fd_validation_result
from validate_database_events import read_sep_validation_results
from validate_maven_events import read_validation_lee_2017
from read_mex_aspera_data import (
    read_mex_ima_bg_counts, 
    clean_up_mex_ima_bg_counts,
    read_aspera_sw_moments)
from read_msl_rad_data import read_msl_rad_doses, read_msl_rad_filtered_e_doses
from read_maven_sep_data import read_maven_sep_flux_data
from read_euhforia_files import read_euhforia_mars_dsv_file



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
    print(df)

    # currentdate = datetime.strptime("2006-01-06", "%Y-%m-%d")
    # startdate = currentdate - pd.Timedelta(days=7)
    # enddate = currentdate + pd.Timedelta(days=7)

    # df = df[
    #    (df["datetime"] > startdate) &
    #    (df["datetime"] < enddate)
    #    ]
    # df_zero = df_zero[(df_zero["datetime"] > startdate) &
    # (df_zero["datetime"] < enddate)
    # ]
    

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["datetime"], df["edac"],
             label='MEX EDAC',
             color=RAW_EDAC_COLOR,
             linestyle = 'dashed',
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
                   color=RAW_EDAC_COLOR)  # RAW_EDAC_COLOR)
    ax2.set_ylabel("Zero-set corrected MEX EDAC count [#]",
                   fontsize=FONTSIZE_AXES_LABELS, color=ZEROSET_COLOR,
                   labelpad=10)  # ZEROSET_COLOR,
    ax1.tick_params(axis="y", labelcolor=RAW_EDAC_COLOR)  # RAW_EDAC_COLOR)
    ax2.tick_params(axis="y", labelcolor=ZEROSET_COLOR)  # ZEROSET_COLOR)
    ax1.minorticks_on()
    ax2.minorticks_on()


    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.set_ylim([-1000, 29000])
    ax2.set_ylim([25500, 36900])
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="upper left",
        fontsize=FONTSIZE_LEGENDS, bbox_to_anchor=(0.1, 1)) 

    #ax1.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    #ax2.legend(loc='upper right',  bbox_to_anchor=(0.9, 1), fontsize=14)
    ax1.grid()
    title = "MEX EDAC counter from Jan 1st, 2004 to Jul 30th, 2024"
    fig.suptitle(title,
                 fontsize=FONTSIZE_TITLE-1)
    plt.tight_layout(pad=1.0)

    plt.savefig('raw_zeroset_corrected_edac_v3.eps',
                dpi=300, transparent=False)
    plt.show()


def plot_rates_only():
    """
    MEX EDAC daily rate only
    """
    df = read_resampled_df()
    print(df)
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

    major_y_locator = MultipleLocator(5)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.set_ylim([0,20])
    # ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax1.grid()
    # fig.suptitle("brat")
    fig.suptitle("MEX EDAC daily rate between Jan. 2004 and Jul. 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.savefig('daily_rate_mex_edac_v3.eps',
                dpi=300, transparent=False)
    plt.show()
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
    plt.suptitle("EDAC GCR component and the sunspot no. between Jan. 2004 and Jul. 2024",
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
    ax1.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    ax2.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS) #bbox_to_anchor=(0.99, 1))
    ax3.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    #ax3.legend(loc='upper left', bbox_to_anchor=(0.05, 1))
    plt.subplots_adjust(hspace=0.1)
    fig.suptitle("MEX EDAC count rates with the solar activity cycle",
                 fontsize=FONTSIZE_TITLE, y=0.99) #0.92
    plt.tight_layout(pad=1.0)
    plt.savefig(
            "rates_all_v6.png",
            dpi=300,
            transparent=False,
        )
    plt.savefig(
            "rates_all_v6.eps",
            dpi=300,
            transparent=False,
        )
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(12, 10))
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
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS, labelpad=0)
    ax1.set_ylabel("EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)
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
    # ax1.minorticks_on()
    # ax3.minorticks_on()
    """
    # ax1.axvline(x=datetime.strptime("2013-12-23", "%Y-%m-%d"),
                linestyle='dashed',
                color='black',
                label='Start of Forbush Decrease')
    """
    ax2.set_ylim([0, 12])
    ax3.set_ylim([-0.5, 10])
    ax3.set_xlim(startdate, enddate)
    #ax2.set_xlim(startdate, enddate)
    #ax1.set_xlim(startdate, enddate)
    major_ticks = pd.date_range(start=startdate+pd.Timedelta(days=1), end=enddate, freq='3D')
    minor_ticks = pd.date_range(start=startdate-pd.Timedelta(days=0), end=enddate, freq='1D')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    #ax3.xaxis.set_major_locator(mdates.DayLocator(14))
    
    #major_ticks_locations = [
    #    pd.to_datetime('2023-02-17 00:00:00')
    #    + pd.Timedelta(days=3 * i)
    #    for i in range(-2, 3)]
    #ax1.set_xticks(major_ticks_locations)
    
    #ax1.xaxis.set_minor_locator(mdates.DayLocator(7))
    #ax3.xaxis.set_minor_locator(mdates.DayLocator(2))
    #ax3.xaxis.set_major_locator(mdates.DayLocator(4))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')
    ax3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')
    # ax3.legend(fontsize=FONTSIZE_LEGENDS,
    # loc='upper right', bbox_to_anchor=(0.9, 1))
    fig.suptitle("MEX EDAC counter at the end of May 2006",
                 fontsize=FONTSIZE_TITLE)
    # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    fig.subplots_adjust(top=0.94)
    plt.savefig('mex_edac_may_2006_nonevent_v2.eps',
                dpi=300, transparent=False)
    plt.savefig('mex_edac_may_2006_nonevent_v2.png',
                dpi=300, transparent=False)
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
    #plt.title("Cumulative Distribution Function")
    #plt.xlabel("Values")
    #plt.ylabel("Cumulative Probability")
    #plt.legend()
    ax1.legend()
    ax1.grid()
    plt.show()


def create_fd_database_plots():
    """
    Plot SWEET for all Forbush decreases
    in the database"""
    df = read_forbush_decreases_database()
    date_list = df["onset_time"].tolist()
    print("Creating plots of all Fd database event dates")
    
    #date_list = df_dates["date"].tolist()
    event_type_list = df["instrument"].tolist()
    folder_name = "fd_database_dates"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_sep_database_plots():
    """Plot SWEET for all SEP dates 
    in the database
    """
    df = read_sep_database_events()
    date_list = df["onset_time"].tolist()
    print("Creating plots of all SEP database event dates")
    
    #date_list = df_dates["date"].tolist()
    event_type_list = df["instrument"].tolist()
    folder_name = "sep_database_dates"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


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


def plot_sweet_events_binned_one_plot():
    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)
    stormy_days_df = read_stormy_sweet_dates()
    print(f"Number of SWEET stormy days is {len(stormy_days_df)}")
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
    #plt.show()


def plot_sweet_events_binned():
    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)
    stormy_days_df = read_stormy_sweet_dates()
    print(f"Number of SWEET stormy days is {len(stormy_days_df)}")
    event_df = read_sweet_event_dates()
    print(event_df)
    sep_df = event_df[event_df["type"] == "SEP"]
    #sep_df = read_sweet_sep_events()
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
    
    plt.savefig(
    "sweet_events_binned.eps",
    dpi=300,
    transparent=False,
        )
    
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
    plt.savefig(
    "sweet_sep_events_binned.eps",
    dpi=300,
    transparent=False,
        )
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
    plt.savefig("sweet_fd_binned.eps",
    dpi=300,
    transparent=False,
    )
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
    suncolor = "#a65628"
    sepcolor = "#EE3377"
    ax1.plot(
        grouped_df["datebin"],
        grouped_df["counts"],
        marker="o",
        color=sepcolor,
        label="Number of SWEET SEP events",
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
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.xaxis.set_major_locator(YearLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(20))

    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax2.set_ylim([0, max(sunspots_smoothed + 15)])
    ax1.set_ylim([0, grouped_df["counts"].max()+2])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Number of SWEET SEP events", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Sunspot number", fontsize=FONTSIZE_AXES_LABELS)
    ax1.tick_params(axis="y", labelcolor=sepcolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)


    ax1.legend(loc="upper left", fontsize=FONTSIZE_LEGENDS)
    ax2.legend(loc="upper right", fontsize=FONTSIZE_LEGENDS)
    plt.title("Sunspot number and number of SEP events in 6-month bins", fontsize=FONTSIZE_TITLE)
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
    print(df)
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
    ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')
    ax1.grid()
    ax1.set_ylim([-25, 600])
    # fig.suptitle("brat")
    fig.suptitle("The sunspot number from January 1810 to January 2025",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.savefig('ssn_all_v3.eps',
                dpi=300, transparent=False)
    plt.savefig('ssn_all_v3.png',
                dpi=300, transparent=False)

    #plt.show()


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
    # raw_edac = read_rawedac()
    raw_edac =  read_zero_set_correct()
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
                                   figsize=(12, 8))
    # ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
    #           label="MEX EDAC", s=3, color=RAW_EDAC_COLOR)
    ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="MEX EDAC", s=3, color=ZEROSET_COLOR)
    #ax1.scatter(df["date"], df["edac_first"],
    #            label="MEX EDAC", color=RAW_EDAC_COLOR)

    ax2.plot(df["date"], df["detrended_rate"], marker="o",
             label="Detrended count rate", color=DETRENDED_EDAC_COLOR)
    
    ax2.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)

    ax1.axvline(x=datetime.strptime("2014-09-02 01:29:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')

    ax2.axvline(x=datetime.strptime("2014-09-02 01:29:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black',
                label='MSL/RAD onset time')
    
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Detrended count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)

    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)
    ax2.tick_params(axis="x", rotation=5)
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position("right")
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    # ax2.axvline(x=datetime.strptime("2014-09-02", "%Y-%m-%d"),
    # linestyle='dashed',
    # color='black')
    ax2.set_ylim([-1, 6])
    ax2.set_ylim([-1, 5])
    ax2.set_xlim(startdate, enddate)
    ax1.set_xlim(startdate, enddate)
    #major_ticks = pd.date_range(start=startdate+pd.Timedelta(days=3), end=enddate, freq='14D')
    #minor_ticks = pd.date_range(start=startdate+pd.Timedelta(days=3), end=enddate, freq='7D')
    #ax1.set_xticks(major_ticks)
    #ax1.set_xticks(minor_ticks, minor=True)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    major_ticks_locations = [
        pd.to_datetime('2014-09-02 00:00:00')
        + pd.Timedelta(days=3* i)
        for i in range(-2, 3)
    ]
    # One week away, up to five weeks
    ax1.set_xticks(major_ticks_locations)
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax1.grid()
    ax2.grid()
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    # ax3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right',
    # bbox_to_anchor=(0.9, 1))
    fig.suptitle("MEX EDAC counter during RAD event in September 2014", fontsize=FONTSIZE_TITLE)
    # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    fig.subplots_adjust(top=0.93, hspace=0.5)



    plt.savefig('mex_edac_sept_2014_v3.eps',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    
    plt.savefig('mex_edac_sept_2014_v3.png',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
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
    print(start_date)
    df = clean_up_mex_ima_bg_counts()
    #df =W read_mex_ima_bg_counts()
    #df = df[df["bg_counts"]<2000]
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
    #print(weeks_in_interval)
    major_ticks_locations = [
        start_date
        + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    # One week away, up to five weeks
    ax.set_xticks(major_ticks_locations)
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=0)
    try:
        ax.set_yscale('log')
        ax.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
        ax.set_ylabel("Counts",  fontsize=FONTSIZE_AXES_LABELS)
        ax.legend()
        ax.set_title("MEX/ASPERA-3 IMA bg. counts", fontsize=FONTSIZE_TITLE,
                    pad=2)
        ax.grid()
        # plt.savefig(LOCAL_DIR / 'events/ima_scan' /
        #            f'ima_{str(start_date.date())}.png',
        #            dpi=300, transparent=False)
        #df.to_csv("temp_ima.txt")
        #sorted = df.sort_values(by='bg_counts').iloc[-40:]
        #print(sorted)
        # sorted.to_csv(LOCAL_DIR / 'events/ima_scan' / f'ima_{str(start_date.date())}.txt',
        #            index = False)
    except ValueError:
        print("failed")
        pass
        
    currentdate = start_date + pd.Timedelta(days=14)
    plt.savefig(LOCAL_DIR / 'ima_events' /
                f'sweet_ima_{str(currentdate.date())}.png',
                dpi=300, transparent=False)
    
    #plt.show()
    plt.close()
                  

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
    sorted = df_ima.sort_values(by="bg_counts")
    print(sorted)
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
    
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_minor_locator(MultipleLocator(5))
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

    # ax3.set_ylim([-1, 8])

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
    df_ima = read_mex_ima_bg_counts()
    print(df_ima)
    df_ima_cleaned = clean_up_mex_ima_bg_counts()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   figsize=(10, 8))
    ax1.plot(df_ima['datetime'], df_ima['bg_counts'], color=IMA_COLOR,
             label='ASPERA-3/IMA background counts')
    ax2.plot(df_ima_cleaned['datetime'], df_ima_cleaned['bg_counts'], color=IMA_COLOR,
             label='ASPERA-3 IMA filtered background counts')
    ax1.set_yscale('log')
    ax1.minorticks_on()
    ax1.xaxis.set_major_locator(YearLocator(4))
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.set_yscale('log')
    ax2.minorticks_on()
    ax2.xaxis.set_major_locator(YearLocator(4))
    ax2.xaxis.set_minor_locator(YearLocator(1))
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Background counts",  fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Background counts", fontsize=FONTSIZE_AXES_LABELS)
    ax1.grid()
    ax2.grid()
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax2.legend(loc='upper right', fontsize=FONTSIZE_LEGENDS)
    ax2.set_ylim([1,7000])
    fig.suptitle('ASPERA-3/IMA background counts 2004-2024', fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.savefig('ima_all_v2.eps',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    plt.savefig('ima_all_v2.png',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    #plt.show()


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
    print(df['speed'].max())
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
    currentdate = datetime.strptime("2011-06-06", "%Y-%m-%d")
    major_ticks_locations = [
        currentdate
        + pd.Timedelta(days=7 * i)
        for i in range(-1, 2)]
    #ax2.set_xticks(major_ticks_locations)
    minor_ticks_locations = [
        currentdate
        + pd.Timedelta(days= i)
        for i in range(-7, 8)]
    #ax2.set_xticks(minor_ticks_locations, minor=True)
    
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
    df_ima = read_mex_ima_bg_counts()
    df_ima = df_ima[(df_ima["datetime"] >= start_time) & (df_ima["datetime"] <= end_time)]
    # df_ima = df_ima[df_ima["bg_counts"]<100]
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
    """
    ax1.axvline(x=datetime.strptime("2022-02-15 22:52:40", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black',
                label='Start of IMA bg. counts enhancement')
    ax2.axvline(x=datetime.strptime("2022-02-15 22:52:40", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black',
                label='Start of IMA bg. counts enhancement')
    """
    ax2.set_xlabel("Time", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("IMA bg. counts", fontsize=FONTSIZE_AXES_LABELS)


    fig.suptitle("MEX EDAC and IMA bg. counts for Feb. 2022 event",
                fontsize=FONTSIZE_TITLE,
                 y=0.95)
    
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(2.5))

    
    ax1.set_yscale('log')
    # ax1.yaxis.tick_right()
    # ax1.yaxis.set_label_position("right")
    # ax2.yaxis.tick_right()
    major_ticks = pd.date_range(start=start_time, end=end_time, freq='2D')
    ax1.set_xticks(major_ticks)

    minor_ticks = pd.date_range(start=start_time, end=end_time, freq='D')
    ax1.set_xticks(minor_ticks, minor=True)

    # ax2.xaxis.set_major_locator(mdates.DayLocator(1))

    #ax2.xaxis.set_minor_locator(mdates.DayLocator(1))
    #ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
    #ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %#d %H'))
    #ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 12)))

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
    """
    if not os.path.exists(LOCAL_DIR / "events/sweet_ima_comparison"):
        os.makedirs(LOCAL_DIR / "events/sweet_ima_comparison")
   
    plt.savefig(LOCAL_DIR / 'events/sweet_ima_comparison' /
                f'sweet_ima_timing{str(start_date.date())}.png',
                dpi=300, transparent=False)
    """

    # plt.close()
    plt.show()
    df_ima['time_difference'] = df_ima['datetime'].diff()
    df_ima['time_difference_in_minutes'] = \
        df_ima['time_difference'].dt.total_seconds() / 60
    

    df_ima.to_csv(LOCAL_DIR / f'ima_data_{start_time.date()}.txt')
    print(df_ima.sort_values(by='bg_counts').iloc[-15:])
    # print(df_ima)


def plot_msl_rad_all():
    df = read_msl_rad_doses()
    print(df)
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
    fig.suptitle("MSL/RAD dose rates from August 2012 to July 2024",
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


    fig.suptitle("MSL/RAD dose rates from August 2012 to July 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    
    plt.subplots_adjust(hspace=0.2)   
    plt.savefig('msl_rad_all_v4.eps',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    plt.savefig('msl_rad_all_v4.png',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    
    # plt.show()


def plot_msl_rad_sweet(start_date, end_date):
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
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS-2)
    ax3.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS-2)
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS-2)
    
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

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
    major_ticks_locations =  [pd.to_datetime('2024-05-20 00:00:00') + pd.Timedelta(days=2 * i) 
                              for i in range(-1, 2)]  # One week away, up to five weeks
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
    

def plot_msl_rad_timerange(start_date, end_date):

    df_rad = read_msl_rad_doses()
    df_rad = df_rad[(df_rad["datetime"] >= start_date) & (df_rad["datetime"] <= end_date)]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='B dose rate', marker='o',
             markersize=2, color=RAD_B_COLOR)
    ax1.plot(df_rad['datetime'], df_rad['E_dose'], label='E dose rate', marker='o',
             markersize=2, color=RAD_E_COLOR)
    #ax1.plot(df_rad['datetime'], df_rad['B_dose'], label='B dose rate', 
    #         color=RAD_B_COLOR)
    #ax1.plot(df_rad['datetime'], df_rad['E_dose'], label='E dose rate',
    #          color=RAD_E_COLOR)

    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.xaxis.set_major_locator(mdates.DayLocator(2))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())

  
    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
        + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]

    major_ticks_locations = [
        pd.to_datetime('2023-08-31 00:00:00')
        + pd.Timedelta(days=7 * i)
        for i in range(-2, 3)]
    #ax1.set_xticks(major_ticks_locations)

    ax1.set_xticks(major_ticks_locations)


    #major_ticks_locations =  [pd.to_datetime('2021-10-28 00:00:00') + pd.Timedelta(days=3 * i) for i in range(-10, 10)]  # One week away, up to five weeks
    #ax3.set_xticks(major_ticks_locations)

    #lower_xlim = df_sweet['date'].iloc[0] - pd.Timedelta(days=0)
    #higher_xlim = df_sweet['date'].iloc[-1] + pd.Timedelta(days=0)
    #ax3.set_xlim(lower_xlim, higher_xlim)
    #ax3.set_ylim(-1, 4)
    
    
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.grid()

    ax1.legend(fontsize=FONTSIZE_LEGENDS)

    fig.suptitle("MSL/RAD during Jan. 2014 event",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    df_rad.to_csv("temp_rad_data.txt")

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
    if not os.path.exists(LOCAL_DIR / "events/sweet_rad_comparison"):
        os.makedirs(LOCAL_DIR / "events/sweet_rad_comparison")

    plt.savefig(LOCAL_DIR / 'events/sweet_rad_comparison' /
                f'timing_sweet_rad_{str(onset_time.date())}.png',
                dpi=300, transparent=False)
    plt.close()


def create_rad_event_plots():
    rad_dates = read_rad_onsets()['onset_time'].tolist()
    for date in rad_dates:
        print(date)
        start_time = pd.to_datetime(date.date()-pd.Timedelta(days=1))
        end_time = pd.to_datetime(date.date()+pd.Timedelta(days=2))-pd.Timedelta(minutes=1)
        print(start_time, end_time)
        plot_rad_sweet_samplewise(start_time, end_time, date)


def plot_filtered_e_doses(start_date, end_date):
    df = read_msl_rad_filtered_e_doses()
    print(df)
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

    fig, ax1 = plt.subplots(figsize=(10, 7))
    #ax1.axvline(x=datetime.strptime("2022-04-04", "%Y-%m-%d"),
    #                    linestyle='dashed',
    #                    color='black',
    #                    label='SWEET SEP event')
    ax1.plot(df['datetime'], df['E_dose'], label='E dose')
    ax1.plot(df['datetime'], df['E_dose_filtered'], label='Filtered E dose', linestyle='dashed')
    plt.show()

# MAVEN/SEP

def test_maven_sep_ion_heatmap(filename):
    print("wo")
    norm = LogNorm(vmin=1, vmax=1e5)
    cmap = 'plasma'
    df = read_maven_sep_flux_data(filename)
    print(df)
    fig, ax = plt.subplots(nrows=1, figsize=(10, 5))


def plot_maven_sep_ion_data_heatmap(filename):
    df = read_maven_sep_flux_data(filename)
    start_date = datetime.strptime("2015-04-15", "%Y-%m-%d")
    #df = df[df['datetime']>=start_date]
    #df = df[["datetime", "20.1-21.5 keV/n,Ion", "21.6-23.0 keV/n,Ion"]]
    df_flux = df.iloc[:,3:31]
    df_datetime = df["datetime"]
    
    df = pd.concat([df_datetime, df_flux], axis=1)
    
    df.set_index('datetime', inplace=True)
    df = df.iloc[:, ::-1]

    print(df.columns)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.T, cmap='plasma', cbar=True, norm=LogNorm(vmin=0.0001, vmax=1e4))
    # daily_ticks = df.index.normalize().drop_duplicates()

    # Select every 7th day
    tick_indices = np.arange(0, len(df.index), 150)
    tick_labels = df.index[tick_indices].date  # Extract corresponding dates

    # Apply to xticks
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=20)
    #plt.xticks(ticks=range(0, len(df.index), len(df) // len(daily_ticks)), labels=daily_ticks.date, rotation=20)
    lower_bounds = [float(col.split('-')[0]) for col in df.columns]
    print(lower_bounds)

    desired_ticks = lower_bounds[::2]
    tick_indices = [i for i, value in enumerate(lower_bounds) if value in desired_ticks]
    tick_labels = [int(value) if value.is_integer() else value for value in desired_ticks]  # Format labels

    # Apply y-ticks
    plt.yticks(ticks=tick_indices, labels=tick_labels, rotation=0)

    #tick_indices = [i for i, value in enumerate(lower_bounds) if value % 100 == 0]
    #tick_labels = [int(lower_bounds[i]) for i in tick_indices]
    #plt.yticks(ticks=tick_indices, labels=tick_labels, rotation=0)
    plt.xlabel('Date')
    plt.ylabel('Ion Energy (keV)')
    plt.title('Heatmap of Energy Flux over Time')
    plt.show()


def plot_maven_sep_fluxes_data_heatmap(filename):
    df = read_maven_sep_flux_data(filename)
    start_date = datetime.strptime("2015-04-15", "%Y-%m-%d")
    end_date = datetime.strptime("2023-07-17", "%Y-%m-%d")
    #df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]
    #df = df[df['datetime']>=start_date]
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
    cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS)  # Set the desired font size for tick labels


    axes[0].set_ylabel('Ion energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    axes[1].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    ion_tick_indices = np.arange(0, len(df_ion.index), 250)
    ion_tick_labels = df_ion.index[ion_tick_indices].date  # Extract corresponding dates

    axes[1].set_xticks(ion_tick_indices)
    axes[1].set_xticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS, 
                            rotation=5)  # Rotate for readability
    #minor_ion_tick_indices = np.arange(0, len(df_ion.index), 50)
    #axes[1].set_xticks(minor_ion_tick_indices, minor=True)
    
    axes[0].set_xlabel('')  # Removes the label
    axes[1].set_xlabel('Date', fontsize=FONTSIZE_AXES_LABELS)
    print(df_ion.columns)
    #ion_lower_bounds = [float(col.split('-')[0]) for col in df_ion.columns]
    ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]
    print("upper boundss")
    print(ion_upper_bounds)
    ion_axis_ticks = ion_upper_bounds[::3]
    ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) if value in ion_axis_ticks]
    ion_tick_labels = [int(value) if value.is_integer() else value for value in ion_axis_ticks]  
    axes[0].set_yticks(ion_tick_indices)
    axes[0].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)

    electron_lower_bounds = [float(col.split('-')[0]) for col in df_electron.columns]
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

    fig.suptitle("MAVEN/SEP hourly fluxes during June 2023 event",
                  fontsize=FONTSIZE_TITLE, y=0.95)
    plt.gca().grid(False)
    plt.show()


def plot_stack_maven_sep_ion_data(filename):
    df = read_maven_sep_flux_data(filename)
    df_flux = df.iloc[:,3:31]
    df_datetime = df["datetime"]
    df = pd.concat([df_datetime, df_flux], axis=1)
    print(df_flux.columns)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    for i in range(1,len(df_flux.columns)+1): #len(df_flux.columns)+1) all
        ax1.plot(df["datetime"], df.iloc[:,i], label=df.columns[i]
                 )
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Ion flux [keV/n]", fontsize=FONTSIZE_AXES_LABELS)

    ax1.set_yscale('log')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1), borderaxespad=0.)
    ax1.grid()
    plt.tight_layout(rect=[0, 0, 1, 1]) 

    plt.show()


def plot_channels_heatmap_below(filename):

    df = read_maven_sep_flux_data(filename)
    start_date = datetime.strptime("2015-04-28", "%Y-%m-%d")
    end_date = datetime.strptime("2015-05-20", "%Y-%m-%d")
    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]
    #df = df[df['datetime']>=start_date]
    df_ion_flux = df.iloc[:,3:31]
    df_electron_flux = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    df_ion.set_index('datetime', inplace=True)
    df_electron.set_index('datetime', inplace=True)
    df_ion = df_ion.iloc[:, ::-1]
    df_electron = df_electron.iloc[:, ::-1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 20), sharex=True)
    heatmap_ion = sns.heatmap(df_ion.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[0])
    sns.heatmap(df_electron.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[1])
    
    axes[1].legend()
    plt.subplots_adjust(hspace=0.05) 
    cbar = fig.colorbar(heatmap_ion.collections[0], ax=axes[0:2], orientation='vertical')
    cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$", 
                   fontsize=FONTSIZE_AXES_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS)  # Set the desired font size for tick labels


    axes[0].set_ylabel('Ion energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    axes[1].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    ion_tick_indices = np.arange(0, len(df_ion.index), 250)
    ion_tick_labels = df_ion.index[ion_tick_indices].date  # Extract corresponding dates

    axes[1].set_xticks(ion_tick_indices)
    axes[1].set_xticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS, 
                            rotation=5)  # Rotate for readability
    #minor_ion_tick_indices = np.arange(0, len(df_ion.index), 50)
    #axes[1].set_xticks(minor_ion_tick_indices, minor=True)
    
    axes[0].set_xlabel('')  # Removes the label
    axes[1].set_xlabel('Date', fontsize=FONTSIZE_AXES_LABELS)
    print(df_ion.columns)
    #ion_lower_bounds = [float(col.split('-')[0]) for col in df_ion.columns]
    ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]
    print("upper boundss")
    print(ion_upper_bounds)
    ion_axis_ticks = ion_upper_bounds[::3]
    ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) if value in ion_axis_ticks]
    ion_tick_labels = [int(value) if value.is_integer() else value for value in ion_axis_ticks]  
    axes[0].set_yticks(ion_tick_indices)
    axes[0].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)

    electron_lower_bounds = [float(col.split('-')[0]) for col in df_electron.columns]
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

    fig.suptitle("MAVEN/SEP hourly fluxes during June 2023 event",
                  fontsize=FONTSIZE_TITLE, y=0.95)
    

    df_flux = df.iloc[:,3:31] #df.iloc[:,3:31]
    df_datetime = df["datetime"]
    df = pd.concat([df_datetime, df_flux], axis=1)
    #print(df_flux.columns)
    channels = [1, 10, 15, 20, -3, -2, -1]
    for elem in channels:
        axes[2].plot(df["datetime"], df.iloc[:,elem], label=df.columns[elem])
    #ax1.plot(df["datetime"], df.iloc[:,10], label=df.columns[10])
    #ax1.plot(df["datetime"], df.iloc[:,27], label=df.columns[27])
    #ax1.plot(df["datetime"], df.iloc[:,-1], label=df.columns[-1])
    #ax1.plot(df["datetime"], df.iloc[:,-2], label=df.columns[-2])
    #ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    axes[2].set_ylabel(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV/n}$"
                   , fontsize=FONTSIZE_AXES_LABELS)

    axes[2].tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    axes[2].tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    #start_date = df_datetime.iloc[0]
    #end_date=df_datetime.iloc[-1]
    #print(start_date, end_date)
    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
    + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    #axes[2].set_xticks(major_ticks_locations)
    #axes[2].xaxis.set_minor_locator(mdates.DayLocator())
    axes[2].set_yscale('log')
    axes[2].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    axes[2].grid()
    axes[2].tick_params(axis="x", rotation=10)
    #plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig.suptitle("MAVEN/SEP hourly fluxes during FTO event #2",
                  fontsize=FONTSIZE_TITLE)
    
    plt.gca().grid(False)
    plt.show()



def plot_channels_heatmap_side_by_side(filename):
    df = read_maven_sep_flux_data(filename)
    current_date = datetime.strptime("2015-05-05", "%Y-%m-%d")
    start_date = current_date - pd.Timedelta(days=7)
    end_date = current_date + pd.Timedelta(days=14)
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

    df_flux = df.iloc[:,3:31] #df.iloc[:,3:31]
    df_datetime = df["datetime"]
    df = pd.concat([df_datetime, df_flux], axis=1)
    #print(df_flux.columns)
    channels = [1, 10, 15, 20, -3, -2, -1]

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2) 
    ax1 = fig.add_subplot(gs[:, 0])
    # fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10,6))
    for elem in channels:
        ax1.plot(df["datetime"], df.iloc[:,elem], label=df.columns[elem])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV/n}$"
                   , fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    #start_date = df_datetime.iloc[0]
    #end_date=df_datetime.iloc[-1]
    #print(start_date, end_date)
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



    df = read_maven_sep_flux_data(filename)
    start_date = datetime.strptime("2015-04-15", "%Y-%m-%d")
    end_date = datetime.strptime("2023-07-17", "%Y-%m-%d")
    #df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]
    #df = df[df['datetime']>=start_date]
    df_ion_flux = df.iloc[:,3:31]
    df_electron_flux = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    df_ion.set_index('datetime', inplace=True)
    df_electron.set_index('datetime', inplace=True)
    df_ion = df_ion.iloc[:, ::-1]
    df_electron = df_electron.iloc[:, ::-1]

    # fig, axes = plt.subplots(2, 1, figsize=(12, 20), sharex=True)
    heatmap_ion = sns.heatmap(df_ion.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[0])
    sns.heatmap(df_electron.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[1])
    
    axes[1].legend()
    plt.subplots_adjust(hspace=0.05) 
    cbar = fig.colorbar(heatmap_ion.collections[0], ax=axes, orientation='vertical')
    cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$", 
                   fontsize=FONTSIZE_AXES_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS)  # Set the desired font size for tick labels


    axes[0].set_ylabel('Ion energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    axes[1].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    ion_tick_indices = np.arange(0, len(df_ion.index), 250)
    ion_tick_labels = df_ion.index[ion_tick_indices].date  # Extract corresponding dates

    axes[1].set_xticks(ion_tick_indices)
    axes[1].set_xticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS, 
                            rotation=5)  # Rotate for readability
    #minor_ion_tick_indices = np.arange(0, len(df_ion.index), 50)
    #axes[1].set_xticks(minor_ion_tick_indices, minor=True)
    
    axes[0].set_xlabel('')  # Removes the label
    axes[1].set_xlabel('Date', fontsize=FONTSIZE_AXES_LABELS)
    print(df_ion.columns)
    #ion_lower_bounds = [float(col.split('-')[0]) for col in df_ion.columns]
    ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]
    print("upper boundss")
    print(ion_upper_bounds)
    ion_axis_ticks = ion_upper_bounds[::3]
    ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) if value in ion_axis_ticks]
    ion_tick_labels = [int(value) if value.is_integer() else value for value in ion_axis_ticks]  
    axes[0].set_yticks(ion_tick_indices)
    axes[0].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)

    electron_lower_bounds = [float(col.split('-')[0]) for col in df_electron.columns]
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

    fig.suptitle("MAVEN/SEP hourly fluxes during June 2023 event",
                  fontsize=FONTSIZE_TITLE, y=0.95)
    plt.gca().grid(False)

    fig.suptitle("MAVEN/SEP hourly fluxes during FTO event #2",
                  fontsize=FONTSIZE_TITLE)
    
    plt.show()


def plot_one_channel_maven_sep_ion_data(filename):
    df = read_maven_sep_flux_data(filename)
    current_date = datetime.strptime("2015-05-13", "%Y-%m-%d")
    start_date = current_date - pd.Timedelta(days=7)
    end_date = current_date + pd.Timedelta(days=7)
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

    df_flux = df.iloc[:,3:31] #df.iloc[:,3:31]
    df_datetime = df["datetime"]
    df = pd.concat([df_datetime, df_flux], axis=1)
    #print(df_flux.columns)
    channels = [1, 10, 15, 20, -3, -2, -1]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for elem in channels:
        ax1.plot(df["datetime"], df.iloc[:,elem], label=df.columns[elem])
    #ax1.plot(df["datetime"], df.iloc[:,10], label=df.columns[10])
    #ax1.plot(df["datetime"], df.iloc[:,27], label=df.columns[27])
    #ax1.plot(df["datetime"], df.iloc[:,-1], label=df.columns[-1])
    #ax1.plot(df["datetime"], df.iloc[:,-2], label=df.columns[-2])
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV/n}$"
                   , fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    #start_date = df_datetime.iloc[0]
    #end_date=df_datetime.iloc[-1]
    #print(start_date, end_date)
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
    fig.suptitle("MAVEN/SEP hourly fluxes during FTO event #2",
                  fontsize=FONTSIZE_TITLE)
    
    plt.show()


def investigate_noise_threshold():
    df = read_detrended_rates()
    df = df[df["detrended_rate"]>= UPPER_THRESHOLD].sort_values(by="detrended_rate")
    target_df = df[["date", "daily_rate", "detrended_rate"]].iloc[0:20]
    target_dates = target_df["date"].tolist()
    count = 1
    for date in target_dates:
        print(date)
        start_time = date - pd.Timedelta(days=3)
        end_time = date + pd.Timedelta(days=3)
        df_raw = read_rawedac()
        df_raw = df_raw[(df_raw["datetime"] >= start_time) & (df_raw["datetime"] <= end_time)]

        df_ima = clean_up_mex_ima_bg_counts()
        df_ima = df_ima[(df_ima["datetime"] >= start_time) & (df_ima["datetime"] <= end_time)]
        df_ima = df_ima[df_ima["bg_counts"]<1000]
        print(len(df_raw["datetime"]), len(df_ima["bg_counts"]))
        if (len(df_raw["datetime"]) > 1) & (len(df_ima["bg_counts"]) > 1):
   
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
            """
            ax1.axvline(x=datetime.strptime("2022-02-15 22:52:40", "%Y-%m-%d %H:%M:%S"),
                        linestyle='dashed',
                        color='black',
                        label='Start of IMA bg. counts enhancement')
            ax2.axvline(x=datetime.strptime("2022-02-15 22:52:40", "%Y-%m-%d %H:%M:%S"),
                        linestyle='dashed',
                        color='black',
                        label='Start of IMA bg. counts enhancement')
            """
            ax2.set_xlabel("Time", fontsize=FONTSIZE_AXES_LABELS)
            ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS)
            ax1.set_ylabel("IMA bg. counts", fontsize=FONTSIZE_AXES_LABELS)

            
            fig.suptitle(f'{date}',
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
            plt.savefig(LOCAL_DIR / 'events/finetuning' /
                    f'{UPPER_THRESHOLD}_{count}_{date.date()}.png',
                    dpi=300, transparent=False)
            plt.close()
            print("figure created")
        count += 1


def plot_real_events_count_rate_distribution():
    """ For the event dates in the SEP database
    plot the distribution of the detrended
    count rates +-1 day of each event
    """

    df = pd.read_csv(TOOLS_OUTPUT_DIR / 
                     'max_detrended_count_rate_for_each_sep_event.txt',
                     skiprows=0, sep="\t",
                     parse_dates=["event_date"])
    print(len(df), "length")
    df = df.dropna()
    data = df["max_detrended_count_rate"]
    mean = data.mean()
    std_dev = data.std()

    max_rate = np.max(data)
    min_rate = np.min(data)
    # bins = np.arange(
    #    min_rate, max_rate + binsize, binsize
    # ) 
    bin_edges = np.arange(int(min_rate)-1, int(max_rate)+1, 0.25)
    print("bin_Edges: ", bin_edges)
    counts, bin_edges = np.histogram(data, bins=bin_edges,
                                     density=False)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bin_edges, color=DETRENDED_EDAC_COLOR,
             edgecolor="black",
             label='Detrended count rate distribution')
    plt.xticks(np.arange(int(min_rate)-2,
                         int(max_rate)+1, 2))
    plt.gca().xaxis.set_minor_locator(
        plt.MultipleLocator(1))

    plt.gca().yaxis.set_minor_locator(
        plt.MultipleLocator(1))
    plt.tick_params(which='major', length=10,
                    labelsize=FONTSIZE_AXES_TICKS)
    plt.tick_params(which='minor', length=6,
                    labelsize=FONTSIZE_AXES_TICKS)
    plt.axvline(x=2.5, label=f'{2.5}',
    color="#EE7733", linestyle='dashed')
    """
    plt.axvline(x=upper_threshold,
                label=f'Detrended count rate = {upper_threshold}',
                color="#EE7733", linestyle='dashed')

    plt.axvline(x=lower_threshold,
                label=f'Detrended count rate = {lower_threshold}',
                color="#EE7733", linestyle='dashed')
    """

    plt.title("Distribution of count rates during verified SEP events",
              fontsize=FONTSIZE_TITLE)
    plt.xlabel("Detrended count rate [#/day]",
               fontsize=FONTSIZE_AXES_LABELS)
    plt.ylabel("Occurrences", fontsize=FONTSIZE_AXES_LABELS)
    plt.legend(fontsize=FONTSIZE_LEGENDS)
    # plt.xlim(-0.5, 8)
    plt.grid()
    plt.show()


def plot_verified_events_rates_distributions():
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)  # row 1, column 2, count 1
    df = pd.read_csv(TOOLS_OUTPUT_DIR / 
                     'max_detrended_count_rate_for_each_sep_event.txt',
                     skiprows=0, sep="\t",
                     parse_dates=["event_date"])
    df = df.dropna()
    data = df["max_detrended_count_rate"]


    sorted = np.sort(data)
    cdf = np.arange(1, len(sorted) + 1) / len(sorted)
    # print(cdf)
    threshold = np.percentile(sorted, 92)
    print(threshold)
    #percentile = percentileofscore(sorted, 2, kind="rank")
    #print(f"VERIFIED: The value {2} is at the {percentile:.2f}th percentile.")

    mean = data.mean()
    std_dev = data.std()

    max_rate = np.max(data)
    min_rate = np.min(data)
    # bins = np.arange(
    #    min_rate, max_rate + binsize, binsize
    # ) 
    bin_edges = np.arange(int(min_rate)-1, int(max_rate)+1, 0.25)
    print("bin_Edges: ", bin_edges)
    counts, bin_edges = np.histogram(data, bins=bin_edges,
                                     density=False)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.hist(data, bins=bin_edges, color=DETRENDED_EDAC_COLOR,
             edgecolor="black",
             label="Detrended EDAC count rate distr.")
    plt.xticks(np.arange(int(min_rate),
                         int(max_rate)+1, 5))
    plt.gca().xaxis.set_minor_locator(
        plt.MultipleLocator(1))

    plt.gca().yaxis.set_minor_locator(
        plt.MultipleLocator(1))
    plt.tick_params(which='major', length=10,
                    labelsize=FONTSIZE_AXES_TICKS)
    plt.tick_params(which='minor', length=6,
                    labelsize=FONTSIZE_AXES_TICKS)
    threshold_color = "#AA4499" #yellow" #"#88CCEE" #"#DDCC77"
    plt.axvline(x=2.5, label=f'Former SWEET noise threshold',
        color=threshold_color, linestyle='dashed')
    """
    plt.axvline(x=upper_threshold,
                label=f'Detrended count rate = {upper_threshold}',
                color="#EE7733", linestyle='dashed')

    plt.axvline(x=lower_threshold,
                label=f'Detrended count rate = {lower_threshold}',
                color="#EE7733", linestyle='dashed')
    """

    plt.title("Histogram of detrended count rates",
              fontsize=FONTSIZE_TITLE-2)
    plt.xlabel("Detrended count rate [#/day]",
               fontsize=FONTSIZE_AXES_LABELS)
    plt.ylabel("Occurrences", fontsize=FONTSIZE_AXES_LABELS)
    plt.legend(fontsize=FONTSIZE_LEGENDS-1)
    # plt.xlim(-0.5, 8)
    plt.grid()

    ##
    ax1 = plt.subplot(1, 2, 2)
    sorted = np.sort(df["max_detrended_count_rate"])
    cdf = np.arange(1, len(sorted) + 1) / len(sorted)
    # print(cdf)
    #threshold = np.percentile(sorted, 92)
    # print(threshold)
    #percentile = percentileofscore(sorted, 2.5, kind="rank")
    #print(f"The value {2.3} is at the {percentile:.2f}th percentile.")

    # fig, ax1 = plt.subplots(figsize=(10, 6))

    plt.title("Cumul. distrib. of detrended count rates",
              fontsize=FONTSIZE_TITLE-2)
    ax1.plot(sorted, cdf, marker='.', 
             linestyle='none', 
             label="CDF of detrended EDAC count rates",
             color=DETRENDED_EDAC_COLOR)
    
    ax1.axvline(2.5,
                color=threshold_color,
                linestyle='dashed',
                label="Former SWEET noise threshold")

    ax1.set_xlabel("Detrended count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Probability", fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)

    ax1.minorticks_on()

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))

    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.grid()
    ax1.legend(fontsize=FONTSIZE_LEGENDS-1)
    plt.tight_layout(pad=1.0)

    plt.savefig('verified_distributions_v2.eps',
                dpi=300, transparent=False)
    
    plt.show()


def plot_cdf_and_histo_detrended_rates():
    
    # using subplot function and creating plot one
    plt.figure(figsize=(14,6))
    plt.subplot(1, 2, 1)  # row 1, column 2, count 1
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
    bin_edges = np.arange(int(min_rate)-1, int(max_rate)+1, 0.25)
    counts, bin_edges = np.histogram(data, bins=bin_edges,
                                     density=False)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
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

    plt.title("Histogram of detrended count rates",
              fontsize=FONTSIZE_TITLE)
    plt.xlabel("Detrended count rate [#/day]",
               fontsize=FONTSIZE_AXES_LABELS)
    plt.ylabel("Occurrences", fontsize=FONTSIZE_AXES_LABELS)
    plt.legend(fontsize=FONTSIZE_LEGENDS)
    plt.xlim(-0.5, 6)
    plt.grid()


    ax1 = plt.subplot(1, 2, 2)
    
    df = read_detrended_rates()
    sorted = np.sort(df["detrended_rate"])
    cdf = np.arange(1, len(sorted) + 1) / len(sorted)
    # print(cdf)
    threshold = np.percentile(sorted, 92)
    print(threshold)
    ##percentile = percentileofscore(sorted, 2.0, kind="rank")
    #print(f"The value {2.0} is at the {percentile:.2f}th percentile.")

    # fig, ax1 = plt.subplots(figsize=(10, 6))

    plt.title("Cumul. distrib. of detrended count rates",
              fontsize=FONTSIZE_TITLE)
    ax1.plot(sorted, cdf, marker='.', 
             linestyle='none', 
             label="CDF of detrended EDAC daily count rates",
             color=DETRENDED_EDAC_COLOR)
    """
    ax1.axhline(
        0.92,
        label="0.92",
        color=THRESHOLD_COLOR
    )
    """
    ax1.set_xlabel("Detrended count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Probability", fontsize=FONTSIZE_AXES_LABELS)

    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)

    ax1.minorticks_on()

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))

    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.grid()

    
    # show plot
    plt.tight_layout(pad=1.0)
    plt.savefig('histo_cdf_edac_v3.eps',
                dpi=300, transparent=False)
    
    plt.show()


def plot_additional_sep_detection():
    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16,10))
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    suncolor = "#a65628"
    sepcolor = "#EE3377"
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], savgolwindow_sunspots, 3
    )
    
    combination_list = [[0, 3], [0, 4], [0, 5], [1, 4], [1.5, 4], [2, 4]]
    for i in range(0, len(combination_list)):
        limit = combination_list[i][0]
        duration = combination_list[i][1]
        df = read_extra_sweet_sep_events(limit, duration)
        df["6_month_group"] = df["date"].apply(group_by_6_months)

        grouped_df = df.groupby("6_month_group").size().reset_index()
        grouped_df.columns = ["datebin", "counts"]
        grouped_df["datebin"] = grouped_df["datebin"] \
            + pd.DateOffset(months=3)
        stormy_total = grouped_df["counts"].sum()
        print(f'Limit: {limit}. Duration: {duration}. Number of SEPs: , {stormy_total}')

        current_ax = axes_list[i]
        current_ax.plot(
            grouped_df["datebin"],
            grouped_df["counts"],
            marker="o",
            color=sepcolor,
            label=f'Limit: {limit}. Min. duration: {duration}',
        )
        # sun_ax = 'ax'+str(i)+str(i)
        sun_ax= current_ax.twinx()
    # ax1.plot(grouped_sep['datebin'],grouped_sep['counts'],
    # marker='o', color=sepcolor,label='SEP',linewidth=0.5,alpha=0.5)
    # ax1.plot(grouped_fd['datebin'],grouped_fd['counts'],
    # marker='o', color=fdcolor, label='FD',linewidth=0.5, alpha=0.5)
    
    # ax2.plot(df_sun['date'], df_sun['daily_sunspotnumber'],
    # label="Number of sunspots")
        sun_ax.plot(
            df_sun["date"],
            sunspots_smoothed,
            linewidth=1,
            color=suncolor,
            label="Smoothed sunspots",
        )
        current_ax.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS-3)
        sun_ax.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS-3)
        current_ax.tick_params(which='minor', length=6)
        sun_ax.tick_params(which='minor', length=6)
        current_ax.xaxis.set_minor_locator(YearLocator(2))
        current_ax.xaxis.set_major_locator(YearLocator(4))
        sun_ax.yaxis.set_minor_locator(MultipleLocator(10))
        sun_ax.yaxis.set_major_locator(MultipleLocator(50))

        current_ax.yaxis.set_minor_locator(MultipleLocator(1))
        current_ax.yaxis.set_major_locator(MultipleLocator(2))
        sun_ax.set_ylim([0, max(sunspots_smoothed + 15)])
        current_ax.set_ylim([0, grouped_df["counts"].max()+2])
        
        #ax1.set_ylabel("Number of SWEET SEP events", fontsize=FONTSIZE_AXES_LABELS)
        #ax11.set_ylabel("Sunspot number", fontsize=FONTSIZE_AXES_LABELS)
        current_ax.tick_params(axis="y", labelcolor=sepcolor)
        if i in [0, 2, 4]:
            #sun_ax.set_yticklabels([])
            pass
        else:
            sun_ax.set_ylabel("Sunspot no.", fontsize=FONTSIZE_AXES_LABELS-2, color=suncolor)
        sun_ax.tick_params(axis="y", labelcolor=suncolor)
        if i in [0, 1, 2, 3]:
            current_ax.set_xticklabels([])
       
    #ax1.legend(loc="upper left", fontsize=FONTSIZE_LEGENDS)
    #ax11.legend(loc="upper right", fontsize=FONTSIZE_LEGENDS)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.9, 1),fontsize=FONTSIZE_LEGENDS)
    ax3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right',bbox_to_anchor=(0.9, 1) )
    ax2.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    ax1.set_ylim([0, 32])
    ax2.set_ylim([0, 26])
    ax3.set_ylim([0, 20])
    ax4.set_ylim([0, 24])
    ax4.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    ax5.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    ax6.legend(loc='upper left', fontsize=FONTSIZE_LEGENDS)
    ax1.yaxis.set_minor_locator(MultipleLocator(3))
    ax1.yaxis.set_major_locator(MultipleLocator(6))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax3.yaxis.set_minor_locator(MultipleLocator(3))
    ax3.yaxis.set_major_locator(MultipleLocator(6))
    ax4.yaxis.set_minor_locator(MultipleLocator(3))
    ax4.yaxis.set_major_locator(MultipleLocator(6))
    ax1.set_ylabel("No. of SEP events", fontsize=FONTSIZE_AXES_LABELS-2, color=sepcolor)
    ax3.set_ylabel("No. of SEP events", fontsize=FONTSIZE_AXES_LABELS-2, color=sepcolor)
    ax5.set_ylabel("No. of SEP events", fontsize=FONTSIZE_AXES_LABELS-2, color=sepcolor)

    ax5.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax6.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    
    fig.suptitle("SWEET SEP events detected with consecutive days method, sorted in 6-month bins",
        y=0.92, fontsize=FONTSIZE_TITLE)
    
    plt.savefig('additional_method_experiment_v3.eps',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    plt.savefig('additional_method_experiment_v3.png',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    
    # plt.show()


from scipy.stats import halfnorm


def half_gaussian_fit():
    df = read_detrended_rates()
    data = df["detrended_rate"]
    params = halfnorm.fit(data)
    # Extract fitted parameters
    loc, scale = params

    print(f"Fitted Parameters: loc={loc}, scale={scale}")

    # Plot the data and the fitted distribution
    x = np.linspace(0, data.max(), 1000)
    pdf = halfnorm.pdf(x, loc=loc, scale=scale)

    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Data Histogram")
    plt.plot(x, pdf, 'r-', label="Fitted Half-Gaussian")
    plt.title("Fitted Half-Gaussian Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def plot_ima_counts_and_sweet_v3(start_date, end_date):
 # df_ima = read_mex_ima_bg_counts()
    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_date) & (df_raw["datetime"] <= end_date)]
    print(df_raw)

    df_ima = clean_up_mex_ima_bg_counts()
    df_ima = df_ima[(df_ima["datetime"] >= start_date) & (df_ima["datetime"] <= end_date)]
    df_sweet = read_detrended_rates()
    df_sweet = df_sweet[(df_sweet["date"] > start_date) & (df_sweet["date"] < end_date)]
    print(df_sweet)
    fig, (ax1, ax2,) = plt.subplots(2, sharex=True,
                                   figsize=(8, 6))
    
    ax1.plot(df_ima["datetime"], df_ima["bg_counts"],
             label="IMA background counts",
             color=IMA_COLOR)
    
    ax4 = ax1.twinx()
    ax4.scatter(df_raw["datetime"], df_raw["edac"],
                label='MEX EDAC counter',
                color= ZEROSET_COLOR,# RAW_EDAC_COLOR, #ZEROSET_COLOR 
                marker='o',
                s=5)
    
    ax1.axvline(x=datetime.strptime("2011-06-04 07:10:00", "%Y-%m-%d %H:%M:%S"),
                        linestyle='dashed',
                        color='black',
                        label='CME detection by STEREO-B')
    ax2.axvline(x=datetime.strptime("2011-06-04 07:10:00", "%Y-%m-%d %H:%M:%S"),
                        linestyle='dashed',
                        color='black')
    #ax0.scatter(df_raw["datetime"], df_raw["edac"],
    #            label='Raw MEX EDAC',
    #            color=RAW_EDAC_COLOR,
    #            linewidth=2,)
   
    
    ax2.plot(df_sweet["date"], df_sweet["detrended_rate"],
             marker='o',
             color=DETRENDED_EDAC_COLOR,
             label='Detrended EDAC count rate')
    
    ax2.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)
    
    
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("Detrended count rate", fontsize=FONTSIZE_AXES_LABELS,
    color=DETRENDED_EDAC_COLOR)
    ax1.set_ylabel("IMA bg. count", fontsize=FONTSIZE_AXES_LABELS,
        color=IMA_COLOR)
    ax4.set_ylabel("MEX EDAC count", fontsize=FONTSIZE_AXES_LABELS,
        color=ZEROSET_COLOR)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)

    fig.suptitle("MEX EDAC and IMA bg. counts for June 2011 event",
                fontsize=FONTSIZE_TITLE,
                 y=0.95)
    #ax2.yaxis.set_major_locator(MultipleLocator(2))
    #ax2.yaxis.set_minor_locator(MultipleLocator(1))

    ax4.yaxis.set_major_locator(MultipleLocator(5))
    ax4.yaxis.set_minor_locator(MultipleLocator(1))

    max_y = df_sweet["detrended_rate"].max()
    ax4.set_ylim([df_raw["edac"].min()-2, df_raw["edac"].max()+2])

    weeks_in_interval = (end_date-start_date).days//7
    major_ticks_locations = [
        start_date
        + pd.Timedelta(days=7 * i)
        for i in range(weeks_in_interval+1)
    ]
    ax2.set_xticks(major_ticks_locations)
    ax2.xaxis.set_minor_locator(mdates.DayLocator())

    ax1.set_yscale('log')
    ax2.tick_params(axis="x", rotation=0)
    ax4.yaxis.tick_left()
    ax4.yaxis.set_label_position("left")

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position("right")

    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax4.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax4.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.tick_params(axis='y') #  labelcolor=DETRENDED_EDAC_COLOR)
    ax1.tick_params(axis='y') # labelcolor=IMA_COLOR
    ax4.tick_params(axis='y')# labelcolor=ZEROSET_COLOR) # RAW_EDAC_COLOR

    ax2.set_ylim([-1, 6])
    #ax1.set_ylim([0.5, 10])

    ax1.grid()
    ax2.grid()
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
    df_ima.to_csv("temp_rate_diff_edac.txt")
    

def plot_euhforia_mars_dsv(filename):
    df = read_euhforia_mars_dsv_file(filename)
    print(df)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 10))


def scan_ima_monthwise():
    currentdate = datetime.strptime("2011-06-06", "%Y-%m-%d")
    date_range = pd.date_range(start="2004-01-01", end="2025-12-31", freq="MS")

    # Convert to a list
    date_list = date_range.to_list()
    for start_date in date_list:
        print(start_date)
        end_date = start_date + pd.Timedelta(days=30)
        plot_mex_ima_bg_counts_time_interval(start_date, end_date)


def plot_solar_wind_boundary():
    # Load colatitude and longitude grid points
    colatitude = np.array([
    1.745329251994309772e-02, 5.235987755982995928e-02, 8.726646259971637676e-02, 
    1.221730476396030163e-01, 1.570796326794896558e-01, 1.919862177193762953e-01, 
    2.268928027592627128e-01, 2.617993877991493523e-01, 2.967059728390359918e-01, 
    3.316125578789226314e-01, 3.665191429188092709e-01, 4.014257279586959104e-01, 
    4.363323129985823279e-01, 4.712388980384689674e-01, 5.061454830783556069e-01, 
    5.410520681182422464e-01, 5.759586531581287749e-01, 6.108652381980153034e-01, 
    6.457718232379019430e-01, 6.806784082777885825e-01, 7.155849933176751110e-01, 
    7.504915783575616395e-01, 7.853981633974482790e-01, 8.203047484373349185e-01, 
    8.552113334772215580e-01, 8.901179185171080865e-01, 9.250245035569946150e-01, 
    9.599310885968812546e-01, 9.948376736367677831e-01, 1.029744258676654312e+00, 
    1.064650843716540951e+00, 1.099557428756427591e+00, 1.134464013796314230e+00, 
    1.169370598836200870e+00, 1.204277183876087287e+00, 1.239183768915973927e+00, 
    1.274090353955860566e+00, 1.308996938995747206e+00, 1.343903524035633623e+00, 
    1.378810109075520263e+00, 1.413716694115406902e+00, 1.448623279155293542e+00, 
    1.483529864195180181e+00, 1.518436449235066599e+00, 1.553343034274953238e+00, 
    1.588249619314839878e+00, 1.623156204354726517e+00, 1.658062789394612935e+00, 
    1.692969374434499574e+00, 1.727875959474386214e+00, 1.762782544514272853e+00, 
    1.797689129554159493e+00, 1.832595714594045910e+00, 1.867502299633932550e+00, 
    1.902408884673819189e+00, 1.937315469713705829e+00, 1.972222054753592246e+00, 
    2.007128639793478886e+00, 2.042035224833365525e+00, 2.076941809873252165e+00, 
    2.111848394913138804e+00, 2.146754979953025444e+00, 2.181661564992912083e+00, 
    2.216568150032798279e+00, 2.251474735072684918e+00, 2.286381320112571558e+00, 
    2.321287905152458197e+00, 2.356194490192344837e+00, 2.391101075232231477e+00, 
    2.426007660272118116e+00, 2.460914245312004311e+00, 2.495820830351891395e+00, 
    2.530727415391777591e+00, 2.565634000431664230e+00, 2.600540585471550870e+00, 
    2.635447170511437509e+00, 2.670353755551324149e+00, 2.705260340591210788e+00, 
    2.740166925631097428e+00, 2.775073510670983623e+00, 2.809980095710870707e+00, 
    2.844886680750756902e+00, 2.879793265790643986e+00, 2.914699850830530181e+00, 
    2.949606435870416821e+00, 2.984513020910303460e+00, 3.019419605950190100e+00, 
    3.054326190990076739e+00, 3.089232776029962935e+00, 3.124139361069850018e+00])

    longitude = np.array([
    -1.207063497903726912e+00, -1.172156912863840272e+00, -1.137250327823953633e+00, 
    -1.102343742784066993e+00, -1.067437157744180576e+00, -1.032530572704293936e+00, 
    -9.976239876644072968e-01, -9.627174026245206573e-01, -9.278108175846341288e-01, 
    -8.929042325447474893e-01, -8.579976475048608497e-01, -8.230910624649742102e-01, 
    -7.881844774250877927e-01, -7.532778923852011532e-01, -7.183713073453145137e-01, 
    -6.834647223054279852e-01, -6.485581372655413457e-01, -6.136515522256548172e-01, 
    -5.787449671857681777e-01, -5.438383821458816492e-01, -5.089317971059950096e-01, 
    -4.740252120661084256e-01, -4.391186270262217861e-01, -4.042120419863352576e-01, 
    -3.693054569464486181e-01, -3.343988719065620341e-01, -2.994922868666754501e-01, 
    -2.645857018267888661e-01, -2.296791167869022821e-01, -1.947725317470156703e-01, 
    -1.598659467071291140e-01, -1.249593616672425161e-01, -9.005277662735593214e-02, 
    -5.514619158746932731e-02, -2.023960654758273983e-02, 1.466697849230385112e-02, 
    4.957356353219044554e-02, 8.448014857207702955e-02, 1.193867336119636136e-01, 
    1.542933186518501976e-01, 1.891999036917368093e-01])

    # Generate synthetic data for heatmap (replace with real data)
    data = np.random.rand(len(colatitude), len(longitude))

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar(label="Value")
    plt.xlabel("Longitude Index")
    plt.ylabel("Colatitude Index")
    plt.title("Heatmap of Colatitude vs Longitude")
    plt.show()


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


def plot_lee_2017_dates():
    print("wa")
    df = read_validation_lee_2017()
    date_list = df["onset_time"].tolist()
    event_type_list = [""]*len(date_list)
    folder_name = "lee_2017_dates"

    create_plots(SWEET_EVENTS_DIR, date_list, folder_name, event_type_list)


def create_stacked_solar_cycle_bins():
    sep_df = read_sweet_sep_events()
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    
    min_date = sep_df["start_date"].min()
    max_date = sep_df["start_date"].max()
    bins = pd.date_range(start=min_date, end=max_date, freq="6M").tolist()
    # Ensure last bin includes max_date even if it's not a full 6-month period
    if bins[-1] < max_date:
        bins.append(max_date)  # Append max_date as the final bin edge


    sep_df["bin"] = pd.cut(sep_df["start_date"], bins=bins, include_lowest=True)

    bins_labels = ["[2.22, 3)", "[3, 4)", "[4, 5)", "[5, ∞)"]
    sep_df["rate_category"] = pd.cut(sep_df["max_rate"], 
                                     bins=[UPPER_THRESHOLD, 3, 4, 5, np.inf], 
                                     right = False, labels=bins_labels)

    sep_df["bin"] = pd.cut(sep_df["start_date"], bins=bins)
    category_counts = sep_df.groupby(["bin", "rate_category"]).size().unstack(fill_value=0)

    category_counts.index = [interval.left for interval in category_counts.index]

    df = category_counts.reset_index()
    df["datetime"] = pd.to_datetime(df["index"])
    df = df[["datetime", "[2.22, 3)", "[3, 4)", "[4, 5)", "[5, ∞)"]]

    df_melted = pd.melt(df, id_vars=['datetime'], value_vars=["[2.22, 3)", "[3, 4)", "[4, 5)", "[5, ∞)"], 
                    var_name='rate_category', value_name='count')
    df_melted['datetime'] = df_melted['datetime'].dt.date
    df = df_melted.pivot(index='datetime', columns='rate_category', values='count')
    df.index = pd.to_datetime(df.index)  # Convert datetime.date to pandas DatetimeIndex
    df.index = df.index.strftime('%Y-%m')  # Converts index to 'YYYY-MM'
    fig, ax1 = plt.subplots(figsize=(10, 6)) 
    ax1.grid()
    df.plot.bar(ax=ax1,stacked=True, width=1, colormap='plasma', edgecolor='black')
    
    if isinstance(df.index, pd.DatetimeIndex):
        #ax1.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years
        #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display only the year
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  #
        # ax1.xaxis.set_major_locator(mdates.AutoDateLocator()) 
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=5)  
    #ax1.grid()
    xticks_positions = np.arange(len(df)) 
    ax1.set_xticks(xticks_positions[::4])  
    ax1.set_xticklabels(df.index[::4], rotation=0)  
    print(df.index[1])
    print(type(df.index[1]))
    print(type(df.index))
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    xticks_minor = np.arange(len(df)) 
    ax1.set_xticks(xticks_minor, minor=True)
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    #ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.tick_params(which='minor', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=16, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(axis="x", rotation=20)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Number of SWEET SEP events", fontsize=FONTSIZE_AXES_LABELS)
    plt.title("Number of SWEET SEP events sorted into 6-month bins", 
              fontsize=FONTSIZE_TITLE)
    plt.legend(fontsize=FONTSIZE_LEGENDS, bbox_to_anchor=(0.95, 1))

    
    plt.savefig('stacked_histo_v3.eps',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    plt.savefig('stacked_histo_v3.png',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    plt.show()

    """
    ax1.plot(
        grouped_sep["datebin"][:-1],
        grouped_sep["counts"][:-1],
        marker="o",
        color=sepcolor,
        label="Number of SWEET SEP events",
        linewidth=1,
        alpha=1,
    )
    #plt.show()
    """
    # plt.close()


def testing():
    data = {
        '2.5': [23, 20, 7, 9, 20, 3],
        '3': [4, 4, 9, 8, 3, 2],
        '4': [0, 2, 2, 0, 1, 0],
        '5': [2, 0, 1, 0, 0, 1]
    }
    index = pd.to_datetime([
        "2004-01-31 12:00:00", "2004-07-31 12:00:00",
        "2005-01-31 12:00:00", "2005-07-31 12:00:00",
        "2006-01-31 12:00:00", "2006-07-31 12:00:00"
    ])
    df = pd.DataFrame(data, index=index)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(ax=ax)  # Plot all columns

    # Set x-ticks every 4 years
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Improve layout
    plt.xticks(rotation=45)
    plt.xlabel("Year")
    plt.ylabel("Values")
    plt.title("Time Series Data")

    plt.show()


def test2():
    data = {
    'datetime': pd.to_datetime([
        '2004-01-31 12:00:00', '2004-07-31 12:00:00', '2005-01-31 12:00:00', '2005-07-31 12:00:00',
        '2006-01-31 12:00:00', '2006-07-31 12:00:00', '2007-01-31 12:00:00', '2007-07-31 12:00:00'
    ]),
    2.5: [23, 20, 7, 9, 20, 3, 5, 9],
    3: [4, 4, 9, 8, 3, 2, 3, 6],
    4: [0, 2, 2, 0, 1, 0, 0, 2],
    5: [2, 0, 1, 0, 0, 1, 0, 0]
}

    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)  # Set datetime as index

    # Convert datetime index to numerical values for proper spacing
    x_values = mdates.date2num(df.index)

    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 100
    # Plot each category as a separate bar, stacked
    bottom = np.zeros(len(df))  # Initialize bottom stack
    for category in [2.5, 3, 4, 5]:  # Iterate through columns
        ax.bar(x_values, df[category], width=bar_width, bottom=bottom, label=f'Rate {category}')
        bottom += df[category]  # Stack the bars

    # Set major ticks at 4-year intervals
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Ensure x-axis labels are readable
    plt.xticks(rotation=45)

    # Labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Stacked Histogram of Rate Categories')
    ax.legend(title='Rate Category')

    plt.show()


def plot_sweet_filtered_e_dose_rate(start_date, end_date):
    df = read_msl_rad_filtered_e_doses()
    print(df)
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

    df_raw = read_rawedac()
    df_raw = df_raw[(df_raw["datetime"] >= start_date) & (df_raw["datetime"] <= end_date)]

    df_sweet = read_detrended_rates()
    df_sweet = df_sweet[(df_sweet["date"] > start_date) & (df_sweet["date"] < end_date)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(10, 6))
    
    ax1.plot(df['datetime'], df['E_dose'], label='E dose rate',
             color=RAD_E_COLOR)
    ax1.plot(df['datetime'], df['E_dose_filtered'], label='Filtered E dose rate', linestyle='dashed',
             color=RAD_B_COLOR)

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
    
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylabel("EDAC counter [#]", fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Dose rate [μGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(5))
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_minor_locator(MultipleLocator(5))
    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    
    #weeks_in_interval = (end_date-start_date).days//4
    #major_ticks_locations = [
    #    start_date
    #    + pd.Timedelta(days=4 * i)
    #    for i in range(weeks_in_interval+4)
    #]

    major_ticks_locations = [
        pd.to_datetime('2022-04-04 00:00:00')
        + pd.Timedelta(days=5 * i)
        for i in range(-6, 5)]

    ax3.set_xticks(major_ticks_locations)
   
    #ax3.xaxis.set_major_locator(mdates.DayLocator(3))
    ax3.xaxis.set_minor_locator(mdates.DayLocator())
    ax3.tick_params(axis="x", rotation=5)
    #major_ticks_locations =  [pd.to_datetime('2021-10-28 00:00:00') + pd.Timedelta(days=3 * i) for i in range(-10, 10)]  # One week away, up to five weeks
    #ax3.set_xticks(major_ticks_locations)

    lower_xlim = df_sweet['date'].iloc[0] - pd.Timedelta(days=2)
    higher_xlim = df_sweet['date'].iloc[-1] + pd.Timedelta(days=2)
    ax3.set_xlim(lower_xlim, higher_xlim)
    ax3.set_ylim(-1, 4)
    ax1.set_ylim(df['E_dose_filtered'].min()-10, df['E_dose_filtered'].max()+20)
    
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
    ax2.legend()
    ax1.legend(loc='upper right', bbox_to_anchor=(0.84, 1))
    ax3.legend()
    fig.suptitle("MSL/RAD and SWEET in March/April 2022",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    #ax1.axvline(x=datetime.strptime("2022-04-04", "%Y-%m-%d"),
    #                    linestyle='dashed',
    #                    color='black',
    #                    label='SWEET SEP event')
    plt.show()


def show_example_sweet_sep_fd():
    # Thesis figure
    startdate = datetime.strptime("2017-08-27", "%Y-%m-%d")
    enddate = datetime.strptime("2017-09-25", "%Y-%m-%d")
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
    major_tick_length=16
    minor_tick_length=10
    fig = plt.figure(figsize=(17, 10))
    subfigs = fig.subfigures(1, 2)  # 1 row, 2 columns
    ax1, ax2, ax3 = subfigs[0].subplots(3, sharex=True, gridspec_kw={"hspace": 0.3})
    bx1, bx2, bx3 = subfigs[1].subplots(3, sharex=True)
    ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="MEX EDAC", s=3, color=ZEROSET_COLOR)
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
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS, labelpad=0)
    ax1.set_ylabel("EDAC count", fontsize=FONTSIZE_AXES_LABELS, labelpad=1)
    ax2.set_ylabel("EDAC count rate", fontsize=FONTSIZE_AXES_LABELS, labelpad=1)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)
    ax3.set_ylabel("Detrended count rate", fontsize=FONTSIZE_AXES_LABELS-4, labelpad=1)
    ax3.tick_params(axis="x", rotation=10)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax3.tick_params(which='minor', length=minor_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=major_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=minor_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=major_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='minor', length=minor_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=major_tick_length, labelsize=FONTSIZE_AXES_TICKS)

    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(5))

    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))

    ax3.yaxis.set_major_locator(MultipleLocator(4))
    ax3.yaxis.set_minor_locator(MultipleLocator(2))
    # ax1.minorticks_on()
    # ax3.minorticks_on()
    
    ax1.axvline(x=datetime.strptime("2017-09-10 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black',
                label='SEP event duration')


    ax1.axvline(x=datetime.strptime("2017-09-13 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')
    
    ax2.axvline(x=datetime.strptime("2017-09-10 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')
    ax3.axvline(x=datetime.strptime("2017-09-10 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')
    ax2.axvline(x=datetime.strptime("2017-09-13 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')

    ax3.axvline(x=datetime.strptime("2017-09-13 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')

    
    #ax2.set_ylim([-1, 6])
    #ax3.set_ylim([-1, 5])
    ax3.set_xlim(startdate-pd.Timedelta(days=2), enddate)
    #ax2.set_xlim(startdate, enddate)
    #ax1.set_xlim(startdate, enddate)
    major_ticks = pd.date_range(start=startdate+pd.Timedelta(days=0), end=enddate, freq='7D')
    minor_ticks = pd.date_range(start=startdate-pd.Timedelta(days=2), end=enddate, freq='1D')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')

    ########### second plot

    # 2015-08-01, 08-30
    startdate = datetime.strptime("2015-08-01", "%Y-%m-%d")
    enddate = datetime.strptime("2015-08-30", "%Y-%m-%d")
    raw_edac = read_zero_set_correct()
    filtered_raw = raw_edac.copy()
    filtered_raw = filtered_raw[
        (filtered_raw["datetime"] > startdate) &
        (filtered_raw["datetime"] < enddate)
    ]

    df = read_detrended_rates()
    df = df[(df["date"] > startdate) & (df["date"] < enddate)]

    bx1.scatter(filtered_raw["datetime"], filtered_raw["edac"], s=3, color=ZEROSET_COLOR)
    #ax1.scatter(df["date"], df["edac_first"],
    #            label="MEX EDAC", color=RAW_EDAC_COLOR)

    bx2.plot(df["date"], df["daily_rate"], marker="o", color=RATE_EDAC_COLOR)

    bx3.plot(df["date"], df["detrended_rate"], marker="o", color=DETRENDED_EDAC_COLOR)

    bx3.axhline(
        UPPER_THRESHOLD, label='Noise threshold',
        linestyle='dashed',
        color=THRESHOLD_COLOR)

    bx1.axvline(x=datetime.strptime("2015-08-11 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black',
                label='FD duration')
    bx2.axvline(x=datetime.strptime("2015-08-11 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')

    bx3.axvline(x=datetime.strptime("2015-08-11 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')

    bx1.axvline(x=datetime.strptime("2015-08-11 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black')

    bx2.axvline(x=datetime.strptime("2015-08-14 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black',)
    bx3.axvline(x=datetime.strptime("2015-08-14 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black'
        )

    bx1.axvline(x=datetime.strptime("2015-08-14 12:00:00", "%Y-%m-%d %H:%M:%S"),
                linestyle='dashed',
                color='black'
        )

    bx3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS, labelpad=0)
    bx1.set_ylabel("EDAC count", fontsize=FONTSIZE_AXES_LABELS, labelpad=1)
    bx2.set_ylabel("EDAC count rate", fontsize=FONTSIZE_AXES_LABELS, labelpad=1)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)
    bx3.set_ylabel("Detrended count rate", fontsize=FONTSIZE_AXES_LABELS, labelpad=1)
    bx3.tick_params(axis="x", rotation=10)
    bx2.yaxis.tick_right()
    bx2.yaxis.set_label_position("right")

    bx3.tick_params(which='minor', length=minor_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    bx3.tick_params(which='major', length=major_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    bx2.tick_params(which='minor', length=minor_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    bx2.tick_params(which='major', length=major_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    bx1.tick_params(which='minor', length=minor_tick_length, labelsize=FONTSIZE_AXES_TICKS)
    bx1.tick_params(which='major', length=major_tick_length, labelsize=FONTSIZE_AXES_TICKS)

    bx1.yaxis.set_major_locator(MultipleLocator(10))
    bx1.yaxis.set_minor_locator(MultipleLocator(5))

    bx2.yaxis.set_major_locator(MultipleLocator(2))
    bx2.yaxis.set_minor_locator(MultipleLocator(1))

    bx3.yaxis.set_major_locator(MultipleLocator(2))
    bx3.yaxis.set_minor_locator(MultipleLocator(1))

    bx3.set_xlim(startdate, enddate)
    #ax2.set_xlim(startdate, enddate)
    #ax1.set_xlim(startdate, enddate)
    major_ticks = pd.date_range(start=startdate+pd.Timedelta(days=3), end=enddate, freq='7D')
    minor_ticks = pd.date_range(start=startdate+pd.Timedelta(days=0), end=enddate, freq='1D')
    bx1.set_xticks(major_ticks)
    bx1.set_xticks(minor_ticks, minor=True)
 
    bx1.grid()
    bx2.grid()
    bx3.grid()
    bx1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    bx2.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right')
    bx3.legend(fontsize=FONTSIZE_LEGENDS, loc='upper right', bbox_to_anchor=(0.9, 1))
    #subfigs[0].tight_layout()  
    subfigs[0].suptitle("Example of a SWEET SEP event", fontsize=FONTSIZE_TITLE, y=0.92)
    subfigs[1].suptitle("Example of a SWEET FD", fontsize=FONTSIZE_TITLE, y=0.92)
    #plt.tight_layout(pad=1)
    plt.subplots_adjust(bottom=0.08)  
    plt.savefig('sweet_events_example_v3.eps',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    
    plt.savefig('sweet_events_example_v3.png',
                dpi=300, transparent=False,
                 bbox_inches="tight", pad_inches=0.05)
    #plt.subplots_adjust(bottom=0.01)  
    plt.show()


if __name__ == "__main__":
    plot_solar_cycle()
    # plot_rates_all()
    # plot_cdf_and_histo_detrended_rates()
    #plot_verified_events_rates_distributions()
    # plot_raw_and_zerosetcorrected()
    # plot_rates_only()
    # plot_gcr_fit_ssn()
    # plot_msl_rad_all()
    # plot_ima_counts_all()
    # create_stacked_solar_cycle_bins()
    #test2()
    #plot_lee_2017_dates() 
    # plot_additional_sep_detection()
    filename = 'maven_f_flux_hr_may_2015'
    #filename = 'maven_f_flux_hr_july_2023'
    #plot_one_channel_maven_sep_ion_data(filename) 
    #filename = 'maven_f_flux_hr_sept_2017'

    #plot_maven_sep_ion_data_heatmap(filename)
    #plot_maven_sep_fluxes_data_heatmap(filename)
    # create_fd_database_plots()
    #create_sep_database_plots()
    
    #plot_stack_maven_sep_ion_data(filename)
    #test_maven_sep_ion_heatmap(filename)
    # plot_one_channel_maven_sep_ion_data(filename) 
    ##plot_channels_heatmap_side_by_side(filename)
    # plot_channels_heatmap_below(filename)
    currentdate = datetime.strptime("2014-09-02", "%Y-%m-%d")
    start_date = currentdate - pd.Timedelta(days=7)
    end_date = currentdate + pd.Timedelta(days=7)
    #show_timerange_counter_countrate(start_date, end_date)
    #plot_msl_rad_sweet(start_date, end_date)
    # plot_msl_rad_sweet(start_date, end_date)
    
    #plot_ima_counts_and_sweet_v3(start_date, end_date)
    #show_timerange_counter_countrate(start_date, end_date)
    # show_timerange(start_date, end_date)
    ima_dates = ["2012-01-27", "2011-06-05", "2024-05-20",
                 "2017-09-10", "2022-02-15", "2012-03-07", "2021-10-28", "2006-12-06",
                 "2024-07-23", "2013-10-11", "2012-05-17", "2005-09-08",
                   "2014-09-10", "2024-01-01", "2014-09-01" ]
    """
    for date in ima_dates:
        currentdate = datetime.strptime(date, "%Y-%m-%d")
        start_date = currentdate - pd.Timedelta(days=14)
        end_date = currentdate + pd.Timedelta(days=14)
        plot_mex_ima_bg_counts_time_interval(start_date, end_date)
    """
    
    start_date = currentdate - pd.Timedelta(days=1)
    end_date = currentdate + pd.Timedelta(days=1)


    currentdate = datetime.strptime("2015-05-05", "%Y-%m-%d")
    start_date = currentdate - pd.Timedelta(days=3)
    end_date = currentdate + pd.Timedelta(days=5)
    #show_timerange(start_date, end_date)

    currentdate = datetime.strptime("2021-10-28", "%Y-%m-%d")
    start_date = currentdate - pd.Timedelta(days=3)
    end_date = currentdate + pd.Timedelta(days=3)
    # plot_ima_sweet_samplewise(start_date, end_date)
    # safe mode maven 1
    start_date = datetime.strptime("2022-02-22", "%Y-%m-%d")
    end_date = datetime.strptime("2022-05-28", "%Y-%m-%d")
    # show_timerange_counter_countrate(start_date, end_date)
    #show_timerange(start_date, end_date)
    # safe mode maven 2

    start_date = datetime.strptime("2023-02-14", "%Y-%m-%d")
    end_date = datetime.strptime("2023-02-28", "%Y-%m-%d")
    #show_timerange_counter_countrate(start_date, end_date)
    
    filename = 'euhforia_Mars.dsv'
    #show_example_sweet_sep_fd()
    # plot_sweet_events_binned()
    #plot_sweet_events_binned_one_plot()
    # plot_additional_sep_det
    # plot_msl_rad_all()
    # plot_raw_and_zerosetcorrected()
    # plot_rates_only()
    #plot_rates_all()
    # plot_cdf_and_histo_detrended_rates()
    #plot_verified_events_rates_distributions()
    #create_stacked_solar_cycle_bins()
    # create_stacked_solar_cycle_bins()
    # plot_ima_counts_all()
    # plot_msl_rad_all()
    # plot_additional_sep_detection()