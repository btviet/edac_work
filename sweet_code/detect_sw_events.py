import os

import pandas as pd
from parameters import FD_NUMBER_DAYS, SWEET_EVENTS_DIR, UPPER_THRESHOLD
from processing_edac import read_resampled_df
from standardize_edac import read_detrended_rates


def find_sep():
    """
    Find the dates in detrended EDAC count rate
    where it is above the upper threshold
    """
    df = read_detrended_rates()
    spike_df = df.copy()
    peaks = spike_df[(spike_df['detrended_rate'] >= UPPER_THRESHOLD)].copy()
    peaks = peaks.sort_values(by='date')

    print("The number of days above the threshold of ",
          UPPER_THRESHOLD, " is: ", len(peaks))
    filename = "sep_dates_edac.txt"
    peaks.to_csv(SWEET_EVENTS_DIR / filename,
                 sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{filename} created")
    # Group the SEP dates into SEP events
    # And find the duration of each SEP event
    # Duration = number of consecutive days above UPPER_THRESHOLD
    peaks['diff'] = peaks['date'].diff().dt.days
    # Identify the sequences where the difference is 1 day or less
    peaks['group'] = (peaks['diff'] != 1).cumsum()

    event_df = peaks.groupby('group').agg(
        start_date=('date', 'first'),
        duration=('diff', lambda x: x.notna().sum()),
        mean_value=('detrended_rate', 'mean')
        # Calculate the mean for the group

    ).reset_index(drop=True)
    event_df.loc[0, 'duration'] = 1

    print(event_df)

    # peaks.loc[:, 'time_difference'] = peaks['date'].diff()
    # print(peaks[peaks['time_difference'] <= pd.Timedelta(days=1)])
    filename = "sep_events_edac.txt"
    # peaks.to_csv(SWEET_EVENTS_DIR / filename,
    #             sep='\t', index=False)  # Save to file
    # print(f"File {SWEET_EVENTS_DIR}/{filename} created")


def find_duration_sep_events():
    event_df = pd.read_csv(SWEET_EVENTS_DIR / 'sep_events_edac.txt',
                           skiprows=0, sep="\t", parse_dates=['date'])
    stormy_df = pd.read_csv(SWEET_EVENTS_DIR / 'sep_dates_edac.txt',
                            skiprows=0, sep="\t", parse_dates=['date'])
    stormy_df['diff'] = stormy_df['date'].diff().dt.days
    print(stormy_df)
    # Identify the sequences where the difference is 1 day or less
    stormy_df['group'] = (stormy_df['diff'] != 1).cumsum()
    print(stormy_df)
    # Group by the sequences and get the first date
    # and the duration (last date - first date + 1 day)
    event_df = stormy_df.groupby('group').agg(
        start_date=('date', 'first'),
        duration=('diff', lambda x: x.notna().sum())
        # Count number of days in the sequence
    ).reset_index(drop=True)
    event_df.loc[0, 'duration'] = 1
    print(event_df)


def add_lenient_seps():
    """
    SEP events occur also when detrended rates are above $0$
    for at least three days
    """
    detrended_df = read_detrended_rates()
    rate_mask = (detrended_df['detrended_rate'] > 0)
    df = pd.DataFrame(rate_mask)
    df["date"] = detrended_df["date"]
    # Group the sequences of Trues and Falses together
    df['group'] = (df['detrended_rate'] !=
                   df['detrended_rate'].shift()).cumsum()
    df = df[df['detrended_rate']]
    df["duration"] = df.groupby('group')['detrended_rate'].transform('size')
    df = df[df["duration"] >= 3]
    df_grouped = df.groupby('group').first().reset_index()
    print(df_grouped)
    file_name = 'extra_sep_days.txt'
    df_grouped[["date", "duration"]].to_csv(
        SWEET_EVENTS_DIR / file_name,
        sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/ {file_name} created")


def read_lenient_sep():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'extra_sep_days.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    sep_df = pd.DataFrame(df[['date', 'duration']],
                          columns=['date', 'duration'])
    sep_df['type'] = 'SEP'
    return sep_df


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

    sep_dates = pd.DataFrame(df[['date', "detrended_rate"]],
                             columns=['date', 'detrended_rate'])
    sep_dates['type'] = 'SEP'
    return sep_dates


def read_all_sep_dates():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'sep_dates_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    sep_dates = pd.DataFrame(df[['date', "detrended_rate"]],
                             columns=['date', "detrended_rate"])
    sep_dates['type'] = 'SEP'
    return sep_dates


def combine_lenient_sep_with_seplist():
    df1 = read_lenient_sep()
    df2 = read_sep_event_df()
    print(df1)
    print(df2)
    df = pd.concat([df1[['date', 'type']], df2[['date', 'type']]])
    print(df)
    return df


def read_fd_df():
    df = pd.read_csv(SWEET_EVENTS_DIR / 'forbush_decreases_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    forbush_dates = pd.DataFrame(df[['date', 'duration']],
                                 columns=['date', 'duration'])
    forbush_dates['type'] = 'Fd'
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


if __name__ == "__main__":
    find_sep()
