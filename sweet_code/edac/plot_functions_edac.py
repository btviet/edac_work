import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DayLocator
from matplotlib.ticker import MultipleLocator

from processing_edac import (read_rawedac, 
                             read_zero_set_correct, 
                             read_resampled_df, 
                             read_rolling_rates)


from sweet_code.parameters import (TOOLS_OUTPUT_DIR,
                                   RAW_EDAC_COLOR, 
                                   ZEROSET_COLOR,
                                   RATE_EDAC_COLOR,
                                   FONTSIZE_LEGENDS,
                                   FONTSIZE_AXES_LABELS, 
                                   FONTSIZE_AXES_TICKS,
                                   FONTSIZE_TITLE)

def plot_raw_edac_scatter(start_date, end_date):
    """
    Plot the MEX EDAC count as a scatterplot.
    For small time intervals
    """
    df = read_rawedac()
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

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
    """
    Plots the MEX EDAC before zero-set correction
    """
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
    fig.suptitle("MEX EDAC Jan 2004 - Aug 2025", fontsize=FONTSIZE_TITLE)

    plt.tight_layout(pad=1.0)
    plt.show()


def plot_zero_set_correction():
    """
    Plots the zero-set corrected EDAC
    """
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
    fig.suptitle("EDAC counter between Jan 2004 and Aug 2025",
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

    ax1.grid()
    title = "MEX EDAC counter from Jan 1st, 2004 to Aug 27th, 2025"
    fig.suptitle(title,
                 fontsize=FONTSIZE_TITLE-1)
    plt.tight_layout(pad=1.0)

    # plt.savefig('raw_zeroset_corrected_edac_v3.eps', dpi=300, transparent=False)
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

    major_y_locator = MultipleLocator(5)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.set_ylim([0,20])
    # ax1.legend(fontsize=FONTSIZE_LEGENDS, loc='upper left')
    ax1.grid()
    fig.suptitle("MEX EDAC daily rate between Jan. 2004 and Aug. 2025",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.savefig('daily_rate_mex_edac_v3.png',
                dpi=300, transparent=False)
    plt.show()
    plt.show()


def plot_rolling_rate():
    df = read_rolling_rates(5)
    rate_mean = round(df['daily_rate'].mean(), 3)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["daily_rate"],
             label=f'MEX EDAC daily rate, mean = {rate_mean} counts per day',
             color='green',
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
    fig.suptitle("MEX EDAC daily rate between Jan 2004 and Apr 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()


def create_large_edac_increments_plots():
    """
    Create plots where the MEX EDAC increments by more than
    2 from one sample to the next.
    The plots should be inspected in case there are data anomalies.
    """
    df = read_rawedac()

    invalid_dates = pd.read_csv(TOOLS_OUTPUT_DIR / "invalid_edac_increases.txt",
                                parse_dates=["datetime"])

    date_list = invalid_dates["datetime"].tolist()
    for date in date_list:
        print(date)
        start_date = date - pd.Timedelta(days=14)
        end_date = date + pd.Timedelta(days=14)
        df_sliced = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.scatter(df_sliced["datetime"], df_sliced["edac"],
                label='Raw MEX EDAC',
                color=RAW_EDAC_COLOR,
                linewidth=2)
        ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
        ax1.set_ylabel("MEX EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
        ax1.tick_params(which='minor', length=6)
        ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=20, ha='right')
        ax1.xaxis.set_minor_locator(DayLocator())
        ax1.legend()
        ax1.grid()
        fig.suptitle(f'{date}', fontsize=FONTSIZE_TITLE)

        plt.tight_layout(pad=1.0)

        if not os.path.exists("large_edac_increments"):
            os.makedirs("large_edac_increments")
        plt.savefig(f'large_edac_increments/{date.date()}.png',
            dpi=300,
            transparent=False
        )
        
        plt.close()

if __name__ == "__main__":
    # plot_raw_edac()
    # plot_zero_set_correction()
    plot_rates_only()
    #create_large_edac_increments_plots()
