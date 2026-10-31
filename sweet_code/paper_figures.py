
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DayLocator
from matplotlib.ticker import MultipleLocator

from sweet.detrend_edac import read_detrended_rates

from edac.processing_edac import (
    read_rawedac,
    read_resampled_df,
    read_rolling_rates,
    read_zero_set_correct,
)

from sweet.detect_sw_events import (
    read_extra_sweet_sep_events,
    read_stormy_sweet_dates,
    read_sweet_event_dates,
    read_sweet_forbush_decreases,
    read_sweet_sep_events,
)


from datetime import datetime

from parameters import (
    RAW_DATA_DIR,
    SUNSPOTS_SAVGOL
)


blue_color = "#4477AA"
cyan_color = "#66CCEE"
green_color = "#228833"
yellow_color = "#CCBB44"
red_color = "#EE6677"
purple_color = "#AA3377"
grey_color = "#BBBBBB"



RAW_EDAC_COLOR = red_color
ZEROSET_COLOR = blue_color
RATE_EDAC_COLOR = green_color
RATE_FIT_COLOR = 'black'
DETRENDED_EDAC_COLOR = green_color
SSN_COLOR = yellow_color
SSN_SMOOTHED_COLOR = 'black'


FONTSIZE_LEGENDS = 8
FONTSIZE_AXES_TICKS = 10
FONTSIZE_AXES_LABELS = 12
FONTSIZE_TITLE = 14

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


def figure_one():

    """
    Plot the raw EDAC counter
    and the zeroset-corrected counter
    in one plot.
    """
    df = read_rawedac()
    df_zero = read_zero_set_correct()
    
    first_date = df_zero["datetime"].iloc[0]
    last_date = df_zero["datetime"].iloc[-1]
    first_edac = df_zero["edac"].iloc[0]
    last_edac = df_zero["edac"].iloc[-1]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["datetime"], df["edac"],
             label='MEX EDAC',
             color=RAW_EDAC_COLOR,
             linestyle = 'dashed',
             linewidth=2.5)
    ax2 = ax1.twinx()
    ax2.plot([first_date, last_date], [first_edac, last_edac],
             label='Straight line between the two endpoints',
             color='black',
             linewidth=2.5,
             linestyle= 'dashed')
    
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
    title = "MEX EDAC counter from Jan. 2004 - Aug. 2025"
    fig.suptitle(title,
                 fontsize=FONTSIZE_TITLE-1)
    plt.tight_layout(pad=1.0)

    #plt.savefig('sweet_code/paper_figures/raw_zeroset_corrected_mex_edac.eps',
    #            dpi=300, transparent=False)
    plt.show()


def figure_two():
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
            "paper_figures/rates_all.eps",
            dpi=300,
            transparent=False,
        )
    plt.show()



def figure_three():
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
    

    #plt.savefig("sweet_binned_v2.eps",
    #dpi=300,
    #transparent=False,
    #    )
    plt.show()

if __name__ == "__main__":

    figure_one()
