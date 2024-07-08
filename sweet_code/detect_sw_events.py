import os

import pandas as pd
from parameters import FD_NUMBER_DAYS, SWEET_EVENTS_DIR, UPPER_THRESHOLD
from processing_edac import read_resampled_df
from standardize_edac import read_standardized_rates


def find_sep():
    """
    Find the dates in standardized EDAC count rate
    where the it is above the upper threshold
    """
    df = read_standardized_rates()
    spike_df = df.copy()
    peaks = spike_df[(spike_df['standardized_rate'] >= UPPER_THRESHOLD)]
    filename = "sep_edac.txt"
    peaks.to_csv(SWEET_EVENTS_DIR / filename,
                 sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{filename} created")


def find_forbush_decreases():
    """
    Find the dates where the EDAC count rate
    is 0 for more than FD_NUMBER_DAYS days"""
    df = read_resampled_df()
    zero_mask = (df['daily_rate'] == 0)
    consecutive_zeros = zero_mask.rolling(window=FD_NUMBER_DAYS).sum()
    rows_with_consecutive_zeros = consecutive_zeros == FD_NUMBER_DAYS
    rows_indices = \
        rows_with_consecutive_zeros[rows_with_consecutive_zeros].index
    result_indices = rows_indices - (FD_NUMBER_DAYS - 1)

    # Return the corresponding rows
    zerodays_df = (df.iloc[result_indices])

    # Dates included in a consecutive zero days sequence
    zerodays_df.loc[:, 'time_difference'] = \
        zerodays_df['date'].diff().fillna(pd.Timedelta(days=1.1))
    # The starting date of each Forbush Decrease
    first_date_df = zerodays_df[zerodays_df['time_difference'] >
                                pd.Timedelta(days=1)]
    zerodays_df.to_csv(SWEET_EVENTS_DIR / 'zerodays.txt',
                       sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/zerodays.txt created")
    first_date_df.to_csv(SWEET_EVENTS_DIR / 'forbush_decreases_edac.txt',
                         sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/forbush_decreases_edac.txt created")


def read_sep_df():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'sep_edac.txt',
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


def merge_sep_fd():
    sep_dates = read_sep_df()
    forbush_dates = read_fd_df()
    spike_df = pd.concat([sep_dates, forbush_dates], ignore_index=True)
    spike_df = spike_df.sort_values(by='date')
    filename = "stormy_dates_edac.txt"
    spike_df.to_csv(SWEET_EVENTS_DIR / filename,
                    sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{filename} created")


def read_forbush_sweet_dates():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'forbush_decreases_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def read_stormy_sweet_dates():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'stormy_dates_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def read_sep_sweet_dates():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'sep_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def detect_edac_events():
    if not os.path.exists(SWEET_EVENTS_DIR):
        os.makedirs(SWEET_EVENTS_DIR)
    find_sep()
    find_forbush_decreases()
    merge_sep_fd()
