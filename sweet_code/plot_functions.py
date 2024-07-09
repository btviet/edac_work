import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detect_sw_events import (
    read_forbush_sweet_dates,
    read_sep_sweet_dates,
    read_stormy_sweet_dates,
)
from old_sweet_comparison import read_validation_results_old
from parameters import (
    DETRENDED_EDAC_COLOR,
    LOCAL_DIR,
    RATE_EDAC_COLOR,
    RATE_FIT_COLOR,
    RAW_DATA_DIR,
    RAW_EDAC_COLOR,
    SSN_COLOR,
    SSN_SMOOTHED_COLOR,
    STANDARDIZED_EDAC_COLOR,
    SUNSPOTS_SAVGOL,
    SWEET_EVENTS_DIR,
    THRESHOLD_COLOR,
    UPPER_THRESHOLD,
)
from processing_edac import read_rawedac
from scipy.signal import savgol_filter
from standardize_edac import read_standardized_rates
from validate_events import read_validation_results


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
    """
    Plot the EDAC count rate,
    the standardized count rate,
    and the solar cycle
    for the entire time period covered by
    MEX EDAC
    """

    standardized_df = read_standardized_rates()

    df_sun = process_sidc_ssn()
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    sunspots_smoothed = savgol_filter(df_sun["daily_sunspotnumber"],
                                      SUNSPOTS_SAVGOL, 3)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))

    ax1.plot(standardized_df["date"],
             standardized_df["daily_rate"], label="Count rate",
             color=RATE_EDAC_COLOR)
    ax1.plot(
        standardized_df["date"],
        standardized_df["gcr_component"],
        label="Savitzky-Golay fit",
    )
    ax2.plot(
        standardized_df["date"],
        standardized_df["standardized_rate"],
        label="Standardized count rate",
        color=STANDARDIZED_EDAC_COLOR
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
    ax3.set_ylabel("Sunspot number")
    ax3.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("EDAC count rate [#/day]", fontsize=10)
    ax2.set_ylabel("Standardized EDAC count rate [#/day]", fontsize=10)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()


def show_timerange(startdate, enddate):
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

    df = read_standardized_rates()

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
    )  # Save selected EDAc rate to file
    edac_change.to_csv(
        LOCAL_DIR / "events" /
        f"EDACrate_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(10, 7))
    ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="Raw EDAC", s=3, color=RAW_EDAC_COLOR)
    ax2.plot(df["date"], df["daily_rate"], marker="o",
             label="EDAC count rate", color=RATE_EDAC_COLOR)

    ax3.plot(df["date"], df["detrended_rate"], marker="o",
             label="De-trended rate", color=DETRENDED_EDAC_COLOR)
    ax3.plot(
        df["date"],
        df["standardized_rate"],
        marker="o",
        label="Standardized EDAC count rate",
        color=STANDARDIZED_EDAC_COLOR
    )
    ax3.axhline(
        UPPER_THRESHOLD, label='threshold', color=THRESHOLD_COLOR
            )
    ax3.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("EDAC count", fontsize=12)
    ax2.set_ylabel("EDAC count rate", fontsize=12)
    # ax3.set_ylabel('EDAC standardized count rate', fontsize=12)
    ax3.set_ylabel("De-trended count rate", fontsize=12)
    ax3.tick_params(axis="x", rotation=20)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.suptitle("De-trending by subtraction", fontsize=16)
    # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    plt.tight_layout(pad=2.0)

    plt.savefig(LOCAL_DIR / 'events' /
                f'edac_{startdate_string}{enddate_string}.png',
                dpi=300, transparent=False)
    plt.show()


def create_plots(file_path, date_list, folder_name):
    """
    Create plots of EDAC count,
    EDAC count rate
    and detrended EDAC count rate
    for a given date_list
    """
    if not os.path.exists(file_path / folder_name):
        os.makedirs(file_path / folder_name)
    raw_edac = read_rawedac()
    df = read_standardized_rates()
    count = 1
    for date in date_list:
        print("Date: ", date)
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
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(8, 7))
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
            label="De-trended rate",
            color=DETRENDED_EDAC_COLOR
        )
        ax3.plot(
            temp_2024["date"],
            temp_2024["standardized_rate"],
            marker="o",
            label="Standardized EDAC count rate",
            color=STANDARDIZED_EDAC_COLOR

        )
        ax3.axhline(UPPER_THRESHOLD, color=THRESHOLD_COLOR)
        ax3.axvline(x=date, color="black", linewidth="1", label=date)
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
        # ax3.legend()
        # plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
        fig.suptitle(str(date.date()), fontsize=16)
        # plt.tight_layout(pad=2.0)
        plt.savefig(
            file_path / folder_name / f"{str(count)}_{date_string}",
            dpi=300,
            transparent=False,
        )

        # plt.show()
        plt.close()
        count += 1


def plot_histogram_rates():
    """
    Plot histogram distribution
    of standardized EDAC count rate"""
    df = read_standardized_rates()
    data = df["standardized_rate"]
    binsize = 0.3
    max_rate = np.max(data)
    min_rate = np.min(data)
    bins = np.arange(
        min_rate, max_rate + binsize, binsize
    )  # Choose the size of the bins

    counts, bin_edges = np.histogram(data, bins=bins,
                                     density=False)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure()
    plt.hist(data, bins=bin_edges, color="#4daf4a", edgecolor="black")
    plt.title("Standardized rate distribution")
    plt.xlabel("Standardized count rate")
    plt.ylabel("Occurrences")
    plt.show()


def group_zerodays(file_path):
    """
    Group how many Forbush decreases
    there has been in
    6 month bins, and plot
    with the solar cycle
    """
    df = read_forbush_sweet_dates(file_path)

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


def create_stormy_plots():
    print("Creating plots of SWEET stormy dates")
    df_dates = read_stormy_sweet_dates()
    date_list = df_dates["date"].tolist()
    folder_name = "stormy_sweet"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name)


def create_fd_plots():
    print("Creating plots of SWEET Forbush decreases")
    df_dates = read_forbush_sweet_dates()
    date_list = df_dates["date"].tolist()
    folder_name = "forbush_decreases_sweet"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name)


def create_sep_plots():
    print("Creating plots of SWEET SEP dates")
    df_dates = read_sep_sweet_dates()
    date_list = df_dates["date"].tolist()
    folder_name = "sep_sweet"
    create_plots(SWEET_EVENTS_DIR, date_list, folder_name)


def plot_stormy_detection():
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df = read_standardized_rates()
    df_sun = process_sidc_ssn()
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], savgolwindow_sunspots, 3
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


def plot_real_eruption_dates():
    """
    Plot the raw EDAC, EDAC count rate and
    the de-trended rate for the CME eruption dates"""
    raw_edac = read_rawedac(
    )
    standardized_df = read_standardized_rates()
    validation_df = read_validation_results()
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
    validation_old = read_validation_results_old()
    validation_new = read_validation_results()
    folder_name = "bothsweets_validation"

    raw_edac = read_rawedac(
    )
    standardized_df = read_standardized_rates()

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
