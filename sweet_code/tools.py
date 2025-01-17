import os
from datetime import datetime

import pandas as pd
from detect_sw_events import read_sweet_sep_dates
from detrend_edac import read_detrended_rates
from parameters import TOOLS_OUTPUT_DIR, UPPER_THRESHOLD
from processing_edac import read_rawedac, read_zero_set_correct
import matplotlib.pyplot as plt
from read_from_database import read_sep_database_events


def find_missing_dates():
    df = read_rawedac()
    print(len(df))
    date_column = df['datetime'].dt.date
    start_date = date_column.iloc[0]
    end_date = date_column.iloc[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    missing_dates = date_range[~date_range.isin(date_column)]
    df = pd.DataFrame(missing_dates, columns=['date'])
    print("Start date is ", start_date, ". End date is ", end_date)
    print("Missing dates: ", missing_dates)
    print(len(missing_dates))

    file_name = "missing_edac_dates.txt"

    if not os.path.exists(TOOLS_OUTPUT_DIR):
        os.makedirs(TOOLS_OUTPUT_DIR)
    df.to_csv(TOOLS_OUTPUT_DIR / file_name,
              sep='\t', index=False)  # Save to file


def read_missing_dates():
    df = pd.read_csv(TOOLS_OUTPUT_DIR / "missing_edac_dates.txt",
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def check_if_date_in_dataset():
    """

    """
    date = datetime.strptime("2022-05-19", "%Y-%m-%d").date()
    df = read_rawedac()
    # df = read_zero_set_correct()
    # df = read_resampled_df()
    dates_in_df = set(df['datetime'].dt.date)
    if date in dates_in_df:
        print(f'{date} in dataframe')


def find_time_interval_in_dataset():
    """
    See how the raw EDAC, detrended EDAC count rate
    looks like in a time interval
    """
    df = read_rawedac()
    startdate = datetime.strptime("2022-05-17", "%Y-%m-%d")
    enddate = datetime.strptime("2022-05-19", "%Y-%m-%d")
    sliced_df = df[(df["datetime"] > startdate) & (df["datetime"] < enddate)]
    print(sliced_df)


def find_sampling_frequency_in_time_interval():
    df = read_rawedac()
    start_date = pd.to_datetime('2024-05-19 12:00:00')
    end_date = pd.to_datetime('2024-05-21 12:00:00')
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

    df['time_difference'] = df['datetime'].diff()
    df['time_difference_in_minutes'] = \
        df['time_difference'].dt.total_seconds() / 60
    
    print(df)

    print("Max time difference: ", df['time_difference_in_minutes'].max())

    if not os.path.exists(TOOLS_OUTPUT_DIR):
        os.makedirs(TOOLS_OUTPUT_DIR)
    df.to_csv(TOOLS_OUTPUT_DIR / f'sampling_frequency_{start_date.date()}_{end_date.date()}.txt',
              sep="\t",
              index=False)


def find_sampling_frequency():
    df = read_rawedac()
    print(df)
    df['time_difference'] = df['datetime'].diff()
    df['time_difference_in_minutes'] = \
        df['time_difference'].dt.total_seconds() / 60
    # Column time_difference is the time between two consecutive samples
    print(df.sort_values(by='time_difference_in_minutes'))
    difference_mean = df['time_difference_in_minutes'].mean()
    print(f"Mean time difference in minutes: {difference_mean}")
    start_date = df['datetime'].iloc[0]
    end_date = df['datetime'].iloc[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    df['date_only'] = df['datetime'].dt.date
    print(df['date_only'].value_counts())
    # number_of_days = len(set(df['datetime'].dt.date))
    print(f"Average number of samples per day: {len(df)/len(date_range)}", )
    df = df.sort_values(by="time_difference_in_minutes")
    grouped_df = df.groupby('time_difference_in_minutes').count()

    bins = [0, 0.5, 1, 5, 10, 60, 24*60]  
    bins = [0, 0.1, 0.5, 1, 5, 10, 60, 24*60]  

    labels = ['0-0.1', '0.1-0.5', '0.5-1', '1-5', '5-10', '10-60', '60-1440']

    df['binned'] = pd.cut(df['time_difference_in_minutes'], bins=bins,
                          labels=labels, right=False)
    # grouped_df = df.groupby('time_difference_in_minutes').count()
    grouped_df = df.groupby('binned').count()
    print("grouped_df: ", grouped_df["datetime"])

    if not os.path.exists(TOOLS_OUTPUT_DIR):
        os.makedirs(TOOLS_OUTPUT_DIR)
    df.to_csv(TOOLS_OUTPUT_DIR / "sampling_frequency.txt",
              sep="\t",
              index=False)

    plt.figure()
    plt.plot(df['datetime'], df['time_difference_in_minutes'])
    plt.show()

    plt.figure()
    plt.hist(df['time_difference_in_minutes'], bins=100)
    plt.xlabel('Time difference in minutes')
    plt.show()


def investigate_stormy_days():
    df = read_sweet_sep_dates()
    sep_rate_mean = df["detrended_rate"].mean()
    print(sep_rate_mean)


def find_last_reading_of_each_day():
    df = read_zero_set_correct()

    df = df.set_index('datetime')
    df = df.groupby(pd.Grouper(freq='D')).last()
    last_df = df.loc[df.groupby(pd.Grouper(freq='D')).idxmax()['edac']]
    print(last_df)


def edac_increments():
    """
    Check if the EDAC counter increases by more than one
    UNFINISHED
    """
    df = read_rawedac()
    df['edac_increment'] = df['edac'].diff()
    df.sort_values(by="edac_increment", inplace=True)
    print(df)


def read_detrended_count_rate_slice():
    df = read_detrended_rates()
    df = df[df["detrended_rate"]>= UPPER_THRESHOLD].sort_values(by="detrended_rate")
    target_df = df[["date", "daily_rate", "detrended_rate"]].iloc[0:20]
    target_df.to_csv(
        TOOLS_OUTPUT_DIR /
        f"{UPPER_THRESHOLD}.csv",
        index=False)

def find_detrended_count_rate_sep_database():
    df_database = read_sep_database_events()
    date_list = df_database["onset_time"].tolist()
    df = read_detrended_rates()
    print(df)
    max_list = []
    for date in date_list:
        print(date.date())
        vicinity_df = df[
            (df["date"].dt.date >= date.date() - pd.Timedelta(days=1)) &
            (df["date"].dt.date <= date.date() + pd.Timedelta(days=1))]
        
        max_detrended_rate = vicinity_df['detrended_rate'].max()
        max_list.append([date, max_detrended_rate])
    max_list_df = pd.DataFrame(max_list)
    max_list_df.columns = ["event_date", "max_detrended_count_rate"]
    max_list_df.to_csv(TOOLS_OUTPUT_DIR / 'max_detrended_count_rate_for_each_sep_event', 
                       sep='\t', index=False)
    print("file saved")


if __name__ == "__main__":
    # edac_increments()
    # find_time_interval_in_dataset()
    # find_sampling_frequency()
    # find_missing_dates()
    # read_missing_dates()
    # find_sampling_frequency()
    # find_sampling_frequency_in_time_interval()
    # find_sampling_frequency()
    # find_last_reading_of_each_day()
    # check_if_date_in_dataset()
    # read_detrended_count_rate_slice()
    read_detrended_count_rate_slice()
