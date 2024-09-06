import os

import pandas as pd
from parameters import FD_NUMBER_DAYS, SWEET_EVENTS_DIR, UPPER_THRESHOLD
from processing_edac import read_resampled_df
from standardize_edac import read_detrended_rates


def find_sep():
    """
    Find the dates in detrended EDAC count rate
    where the it is above the upper threshold
    """
    df = read_detrended_rates()
    spike_df = df.copy()
    peaks = spike_df[(spike_df['detrended_rate'] >= UPPER_THRESHOLD)].copy()

    # print("Peaks: ", peaks)
    # print("upper threshold: ", UPPER_THRESHOLD)
    print("The number of days above the threshold of ",
          UPPER_THRESHOLD, " is: ", len(peaks))
    filename = "sep_dates_edac.txt"
    peaks.to_csv(SWEET_EVENTS_DIR / filename,
                 sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{filename} created")

    # Group the SEP dates into SEP events
    peaks.loc[:, 'time_difference'] = peaks['date'].diff()

    peaks = peaks[peaks['time_difference'] > pd.Timedelta(days=1)]
    peaks = peaks.sort_values(by="time_difference")

    filename = "sep_events_edac.txt"
    peaks.to_csv(SWEET_EVENTS_DIR / filename,
                 sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{filename} created")


def find_forbush_decreases():
    """
    Find the dates where the EDAC count rate
    is 0 for more than FD_NUMBER_DAYS days
    """
    resampled_df = read_resampled_df()

    zero_mask = (resampled_df['daily_rate'] == 0)
    df = pd.DataFrame(zero_mask)
    df.rename(columns={'daily_rate': 'zero_rate'}, inplace=True)
    df["date"] = resampled_df["date"]
    # Group the sequences of Trues and Falses together
    df['group'] = (df['zero_rate'] != df['zero_rate'].shift()).cumsum()
    # Keep the groups which have zero rates
    df = df[df['zero_rate']]

    df["duration"] = df.groupby('group')['zero_rate'].transform('size')
    # Keep only the days that are 0 for at least FD_NUMBER_DAYS days
    df = df[df["duration"] >= FD_NUMBER_DAYS]
    df[["date", "group"]].to_csv(SWEET_EVENTS_DIR / 'zerodays.txt',
                                 sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/zerodays.txt created")
    # Keep only the first dates in each Forbush decrease
    df_grouped = df.groupby('group').first().reset_index()

    df_grouped[["date", "duration"]].to_csv(
        SWEET_EVENTS_DIR / 'forbush_decreases_edac.txt',
        sep='\t', index=False)  # Save to file

    print(f"File {SWEET_EVENTS_DIR}/forbush_decreases_edac.txt created")


def read_sep_event_df():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'sep_events_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    sep_dates = pd.DataFrame(df['date'],
                             columns=['date'])
    sep_dates['type'] = 'SEP'
    return sep_dates


def read_all_sep_dates():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'sep_dates_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    sep_dates = pd.DataFrame(df['date'],
                             columns=['date'])
    sep_dates['type'] = 'SEP'
    return sep_dates


def read_fd_df():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'forbush_decreases_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    forbush_dates = pd.DataFrame(df['date'], columns=['date'])
    forbush_dates['type'] = 'Forbush'
    return forbush_dates


def read_zero_df():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'zerodays.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    forbush_dates = pd.DataFrame(df['date'], columns=['date'])
    forbush_dates['type'] = 'Forbush'
    return forbush_dates


def create_stormy_days_list():
    sep_dates = read_all_sep_dates()
    forbush_dates = read_zero_df()
    spike_df = pd.concat([sep_dates, forbush_dates], ignore_index=True)
    spike_df = spike_df.sort_values(by='date')
    filename = "stormy_dates_edac.txt"
    spike_df.to_csv(SWEET_EVENTS_DIR / filename,
                    sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{filename} created")


def create_sw_event_list():
    sep_events = read_sep_event_df()
    forbush_decreases = read_fd_df()
    event_df = pd.concat([sep_events, forbush_decreases], ignore_index=True)
    event_df = event_df.sort_values(by="date")
    filename = "sweet_events.txt"
    event_df.to_csv(SWEET_EVENTS_DIR / filename,
                    sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{filename} created")


def read_stormy_sweet_dates():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'stormy_dates_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def read_sweet_event_dates():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'sweet_events.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])

    return df


def detect_edac_events():
    if not os.path.exists(SWEET_EVENTS_DIR):
        os.makedirs(SWEET_EVENTS_DIR)
    find_sep()
    find_forbush_decreases()
    create_stormy_days_list()
    create_sw_event_list()
