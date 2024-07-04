import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detect_sw_events import read_forbush_dates, read_stormy_dates
from parameters import SUNSPOTS_SAVGOL, UPPER_THRESHOLD
from processing_edac import read_rawedac
from scipy.signal import savgol_filter
from standardize_edac import read_standardized_rates


def process_sidc_ssn(file_path):
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
    df_sun = pd.read_csv(file_path / "SN_d_tot_V2.0.csv",
                         names=column_names, sep=";")
    df_sun = df_sun[df_sun["daily_sunspotnumber"] >= 0]

    df_sun["date"] = pd.to_datetime(df_sun[["year", "month", "day"]])
    df_sun = df_sun[["date", "daily_sunspotnumber"]]
    return df_sun


def plot_rates_all(file_path):
    """
    Plots the EDAC count rate,
    the standardized count rate,
    and the solar cycle
    for the entire time period
    """

    standardized_df = read_standardized_rates(file_path)

    df_sun = process_sidc_ssn(file_path)
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    sunspots_smoothed = savgol_filter(df_sun["daily_sunspotnumber"],
                                      SUNSPOTS_SAVGOL, 3)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))

    ax1.plot(standardized_df["date"],
             standardized_df["daily_rate"], label="Count rate")
    ax1.plot(
        standardized_df["date"],
        standardized_df["gcr_component"],
        label="Savitzky-Golay fit",
    )
    ax2.plot(
        standardized_df["date"],
        standardized_df["standardized_rate"],
        color="#4daf4a",
        label="Standardized count rate",
    )
    ax3.plot(
        df_sun["date"],
        df_sun["daily_sunspotnumber"],
        color="#f781bf",
        label="Sunspot number",
    )
    ax3.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color="#a65628",
        label="Smoothed sunspot number",
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


def show_timerange(file_path, startdate, enddate):
    raw_edac = read_rawedac(file_path)
    filtered_raw = raw_edac.copy()
    filtered_raw = filtered_raw[
        (filtered_raw["datetime"] > startdate) &
        (filtered_raw["datetime"] < enddate)
    ]

    df = read_standardized_rates(file_path)  # If SUBTRACTION method

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
    if not os.path.exists(file_path / "events"):
        os.makedirs(file_path / "events")

    filtered_raw.to_csv(
        file_path / "events" /
        f"rawEDAC_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )  # Save selected raw EDAC to file
    df.to_csv(
        file_path / "events" /
        f"EDACrate_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )  # Save selected EDAc rate to file
    edac_change.to_csv(
        file_path / "events" /
        f"EDACrate_{startdate_string}-{enddate_string}.txt",
        sep="\t",
        index=False,
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(10, 7))
    ax1.scatter(filtered_raw["datetime"], filtered_raw["edac"],
                label="Raw EDAC", s=3)
    ax2.plot(df["date"], df["daily_rate"], marker="o",
             label="EDAC count rate")

    ax3.plot(df["date"], df["detrended_rate"], marker="o",
             label="De-trended rate")
    ax3.plot(
        df["date"],
        df["standardized_rate"],
        marker="o",
        color="#4daf4a",
        label="Standardized EDAC count rate",
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

    # plt.savefig(file_path/'events'
    # /f'edac_{startdate_string}{enddate_string}.png',
    # dpi=300, transparent=False)
    plt.show()


def create_plots(file_path, date_list, folder_name):
    """
    Create plots of EDAC count,
    EDAC count rate
    and detrended EDAC count rate
    """
    raw_edac = read_rawedac(file_path)
    df = read_standardized_rates(file_path)
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
                    label="Raw EDAC", s=3)
        ax2.plot(
            temp_2024["date"],
            temp_2024["daily_rate"],
            marker="o",
            label="EDAC count rate",
        )

        ax3.plot(
            temp_2024["date"],
            temp_2024["detrended_rate"],
            marker="o",
            label="De-trended rate",
        )
        ax3.plot(
            temp_2024["date"],
            temp_2024["standardized_rate"],
            marker="o",
            color="#4daf4a",
            label="Standardized EDAC count rate",
        )
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

        if not os.path.exists(file_path / folder_name):
            os.makedirs(file_path / folder_name)
        plt.savefig(
            file_path / folder_name / f"{str(count)}_{date_string}",
            dpi=300,
            transparent=False,
        )

        # plt.show()
        plt.close()
        count += 1


def plot_histogram_rates(file_path):

    df = read_standardized_rates(file_path)
    data = df["standardized_rate"]  # If subtraction method
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
    df = read_forbush_dates(file_path)

    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)

    df["6_month_group"] = df["date"].apply(group_by_6_months)
    grouped_df = df.groupby("6_month_group").size().reset_index()
    grouped_df.columns = ["datebin", "counts"]
    grouped_df["datebin"] = grouped_df["datebin"] \
        + pd.DateOffset(months=3)
    stormy_total = grouped_df["counts"].sum()
    print("number of forbush decreases: ", stormy_total)

    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df_sun = process_sidc_ssn(file_path)
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


def create_fd_plots(file_path):
    df_dates = read_forbush_dates(file_path)
    date_list = df_dates["date"].tolist()
    folder_name = ""
    create_plots(file_path, date_list, folder_name)


def create_sep_plots(file_path):
    df_dates = read_stormy_dates()
    date_list = df_dates["date"].tolist()
    folder_name = "stormy"
    create_plots(file_path, date_list, folder_name)


def plot_stormy_detection(file_path):
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df = read_standardized_rates(file_path)
    df_sun = process_sidc_ssn(file_path)
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed = savgol_filter(
        df_sun["daily_sunspotnumber"], savgolwindow_sunspots, 3
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))

    ax1.plot(df["date"], df["daily_rate"], label="EDAC count rate")
    ax1.plot(df["date"], df["gcr_component"], label="Savitzky-Golay fit")

    ax2.plot(
        df["date"],
        df["standardized_rate"],
        color="#4daf4a",
        label="Standardized count rate",
    )
    ax2.axhline(
        UPPER_THRESHOLD, color="purple",
        label="Threshold: " + str(UPPER_THRESHOLD)
    )
    # ax2.axhline(lower_threshold, color= thresholdcolor,
    #  label='Threshold: ' + str(lower_threshold))
    ax3.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Count rate [#/day]", fontsize=10)
    ax2.set_ylabel("Standardized count rate [#/day]", fontsize=10)
    ax3.plot(
        df_sun["date"],
        df_sun["daily_sunspotnumber"],
        color="#f781bf",
        label="Number of sunspots",
    )
    ax3.plot(
        df_sun["date"],
        sunspots_smoothed,
        linewidth=1,
        color="#a65628",
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


def plot_stormy_days_bin(file_path):
    def group_by_6_months(date):
        return pd.Timestamp(date.year, ((date.month - 1) // 6) * 6 + 1, 1)

    spike_df = read_stormy_dates(file_path)
    sep_df = spike_df[spike_df["type"] == "SEP"]
    forbush_df = spike_df[spike_df["type"] == "Forbush"]
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    df_sun = process_sidc_ssn(file_path)
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
