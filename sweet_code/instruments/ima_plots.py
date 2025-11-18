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
from sweet_code.instruments.read_mex_aspera_data import clean_up_mex_ima_bg_counts
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

def plot_mex_ima_bg_counts_time_interval(start_date, end_date):
    print(start_date)
    df = clean_up_mex_ima_bg_counts()
    #df =W read_mex_ima_bg_counts()
    #df = df[df["bg_counts"]<2000]
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    print(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['datetime'], df['bg_counts'],
               label="Background counts",
               marker='o')
    # ax.scatter(df['datetime'], df['total_counts'], s=0.5,
    #           label="Total counts")

    ax.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    # weeks_in_interval = (end_date-start_date).days//7
    # print(weeks_in_interval)
    #major_ticks_locations = [
    #    start_date
    #    + pd.Timedelta(days=7 * i)
    #    for i in range(weeks_in_interval+1)
    #]
    # One week away, up to five weeks
    #ax.set_xticks(major_ticks_locations)

    major_ticks_locations =  [onset_time.date() + pd.Timedelta(days=2 * i) 
                              for i in range(-2, 4)]  
    major_ticks_locations =  [onset_time.date() + pd.Timedelta(days=5 * i) 
                              for i in range(-2, 4)]  
    ax.set_xticks(major_ticks_locations)

    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=0)

    date_format = mdates.DateFormatter('%b %d')  # '%b' = abbreviated month name, '%d' = day of month
    ax.xaxis.set_major_formatter(date_format)

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
    # plt.savefig(LOCAL_DIR / 'ima_events' /
    #            f'sweet_ima_{str(currentdate.date())}.png',
    #            dpi=300, transparent=False)
    
    plt.show()
    # plt.close()
                  
if __name__ == "__main__":

    onset_time = datetime.strptime("2024-10-09 17:48", "%Y-%m-%d %H:%M")
    start_time = onset_time - pd.Timedelta(days=14)
    end_time = onset_time + pd.Timedelta(days=14)

    plot_mex_ima_bg_counts_time_interval(start_time, end_time)