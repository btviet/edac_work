import pandas as pd
from file_paths import get_data_dir
from parameters import UPPER_THRESHOLD
from processing_edac import read_resampled_df
from standardize_edac import read_standardized_rates


def find_sep(file_path):
    df = read_standardized_rates(file_path)
    upper_threshold = UPPER_THRESHOLD
    spike_df = df.copy()
    peaks = spike_df[(spike_df['standardized_rate'] >= upper_threshold)]
    peaks.to_csv(file_path / 'sep.txt', sep='\t', index=False)  # Save to file


def find_forbush_decreases(file_path):
    df = read_resampled_df(file_path)
    zero_mask = (df['daily_rate'] == 0)
    num_zeros = 3
    consecutive_zeros = zero_mask.rolling(window=num_zeros).sum()
    rows_with_consecutive_zeros = consecutive_zeros == num_zeros
    rows_indices = \
        rows_with_consecutive_zeros[rows_with_consecutive_zeros].index
    result_indices = rows_indices - (num_zeros - 1)

    # Return the corresponding rows
    zerodays_df = (df.iloc[result_indices])

    # Dates included in a consecutive zero days sequence
    zerodays_df.loc[:, 'time_difference'] = \
        zerodays_df['date'].diff().fillna(pd.Timedelta(days=1.1))
    # The starting date of each Forbush Decrease
    first_date_df = zerodays_df[zerodays_df['time_difference'] >
                                pd.Timedelta(days=1)]

    zerodays_df.to_csv(file_path / 'zerodays.txt',
                       sep='\t', index=False)  # Save to file
    first_date_df.to_csv(file_path / 'forbush_decreases.txt',
                         sep='\t', index=False)  # Save to file


def read_sep_df(file_path):
    df = pd.read_csv(file_path / 'sep.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    sep_dates = pd.DataFrame(df['date'],
                             columns=['date'])
    sep_dates['type'] = 'SEP'
    return sep_dates


def read_fd_df(file_path):
    df = pd.read_csv(file_path / 'forbush_decreases.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    forbush_dates = pd.DataFrame(df['date'], columns=['date'])
    forbush_dates['type'] = 'Forbush'
    return forbush_dates


def merge_sep_fd(file_path):
    sep_dates = read_sep_df(file_path)
    forbush_dates = read_fd_df(file_path)
    spike_df = pd.concat([sep_dates, forbush_dates], ignore_index=True)
    spike_df = spike_df.sort_values(by='date')
    spike_df.to_csv(file_path / 'stormy_dates.txt',
                    sep='\t', index=False)  # Save to file
    print("stormy_dates.txt created")


def read_forbush_dates(file_path):
    df = pd.read_csv(file_path / 'forbush_decreases.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def read_stormy_dates(file_path):
    df = pd.read_csv(file_path / 'stormy_dates.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def detect_edac_events():
    file_path = get_data_dir()
    find_sep(file_path)
    find_forbush_decreases(file_path)
    merge_sep_fd(file_path)
