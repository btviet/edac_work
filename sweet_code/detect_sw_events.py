import os

import pandas as pd
from detrend_edac import read_detrended_rates
from parameters import (
    FD_NUMBER_DAYS,
    LOWER_THRESHOLD,
    SWEET_EVENTS_DIR,
    UPPER_THRESHOLD,
    TOOLS_OUTPUT_DIR
)
from processing_edac import read_resampled_df

sep_dates_filename = 'sweet_sep_dates.txt'
sep_events_filename = f'sep_events_sweet_{UPPER_THRESHOLD}.txt'
extra_sep_days_filename = 'extra_sweet_sep_days.txt'
extra_sep_events_filename = 'extra_sweet_sep_events.txt'
zerodays_filename = 'sweet_zerodays.txt'
forbush_decrease_events_filename = 'sweet_forbush_decreases.txt'
sweet_events_filename = 'sweet_events.txt'
stormy_days_filename = 'sweet_stormy_days.txt'


def find_sweet_sep():
    """
    Find the dates where detrended EDAC count rate
    exceeds the UPPER_THRESHOLD.
    Group them into SEP events with the duration
    (duration = number of consecutive days above UPPER_THRESHOLD)
    and mean value
    """
    print("----- Finding SEP events in the EDAC data --------")
    df = read_detrended_rates()
    spike_df = df.copy()


    peaks = spike_df[(spike_df['detrended_rate'] > UPPER_THRESHOLD)].copy()
    # Remove invalid dates
    invalid_dates = pd.read_csv(TOOLS_OUTPUT_DIR / "invalid_edac_increases.txt",
                                parse_dates=["datetime"])

    date_list = invalid_dates["datetime"].dt.date.tolist()
    peaks = peaks.sort_values(by='date')
    peaks = peaks[~peaks["date"].dt.date.isin(date_list)]


    print("The number of days above the threshold of ",
          UPPER_THRESHOLD, " is: ", len(peaks))
    peaks.to_csv(SWEET_EVENTS_DIR / sep_dates_filename,
                 sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\{sep_dates_filename} created")
    # Group the SEP dates into SEP events
    # And find the duration of each SEP event
    # Duration = number of consecutive days above UPPER_THRESHOLD
    peaks['diff'] = peaks['date'].diff().dt.days
    # Identify the sequences where the difference is 1 day or less
    peaks['group'] = (peaks['diff'] != 1).cumsum()

    event_df = peaks.groupby('group').agg(
        start_date=('date', 'first'),
        duration=('diff', lambda x: x.notna().sum()),
        mean_value=('detrended_rate', 'mean'),
        max_value=('detrended_rate', 'max')

    ).reset_index(drop=True)
    event_df.loc[0, 'duration'] = 1
    event_df.rename(columns={'mean_value': 'mean_rate',
                             'max_value': 'max_rate'}, inplace=True)
    print(f'Number of SWEET SEP events is {len(event_df)}')
    event_df.to_csv(SWEET_EVENTS_DIR / sep_events_filename,
                    sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\{sep_events_filename} created")

    # peaks.loc[:, 'time_difference'] = peaks['date'].diff()
    # print(peaks[peaks['time_difference'] <= pd.Timedelta(days=1)])

    # peaks.to_csv(SWEET_EVENTS_DIR / filename,
    #             sep='\t', index=False)  # Save to file
    # print(f"File {SWEET_EVENTS_DIR}/{filename} created")


def test_sweet_sep_variable_threshold():
    print("----- Finding SEP events in the EDAC data --------")
    print("----- testing")
    df = read_detrended_rates()
    
    spike_df = df.copy()
    spike_df["threshold"] = df["gcr_component"]+1
    print(spike_df["threshold"].min(), spike_df["threshold"].max())
    peaks = spike_df[(spike_df['detrended_rate'] > spike_df["threshold"])].copy()
    peaks = peaks.sort_values(by='date')
    
    print("The number of days above the threshold of ",
          UPPER_THRESHOLD, " is: ", len(peaks))
    peaks.to_csv(SWEET_EVENTS_DIR / sep_dates_filename,
                 sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\{sep_dates_filename} created")
    # Group the SEP dates into SEP events
    # And find the duration of each SEP event
    # Duration = number of consecutive days above UPPER_THRESHOLD
    peaks['diff'] = peaks['date'].diff().dt.days
    # Identify the sequences where the difference is 1 day or less
    peaks['group'] = (peaks['diff'] != 1).cumsum()

    event_df = peaks.groupby('group').agg(
        start_date=('date', 'first'),
        duration=('diff', lambda x: x.notna().sum()),
        mean_value=('detrended_rate', 'mean'),
        max_value=('detrended_rate', 'max')

    ).reset_index(drop=True)
    event_df.loc[0, 'duration'] = 1
    event_df.rename(columns={'mean_value': 'mean_rate',
                             'max_value': 'max_rate'}, inplace=True)
    print(f'Number of SWEET SEP events is {len(event_df)}')
    event_df.to_csv(SWEET_EVENTS_DIR / sep_events_filename,
                    sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\{sep_events_filename} created")


def find_sweet_sep_by_consecutive_days():
    """
    Find SEP events by finding the dates where
    the detrended rates are above $0$
    for at least three days
    """
    print("----- Finding SEP events (second method) \
          in the EDAC data --------")
    detrended_df = read_detrended_rates()
    rate_mask = (detrended_df['detrended_rate'] > 0)
    df = pd.DataFrame(rate_mask)
    df["date"] = detrended_df["date"]
    # Group the sequences of Trues and Falses together
    df['group'] = (df['detrended_rate'] !=
                   df['detrended_rate'].shift()).cumsum()
    df = df[df['detrended_rate']]
    df["duration"] = df.groupby('group')['detrended_rate'].transform('size')

    df = df[df["duration"] >= 4]
    print(f'The number of extra SEP days: {len(df)}')
    df.to_csv(SWEET_EVENTS_DIR / extra_sep_days_filename,
              sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\{extra_sep_days_filename} created")

    df_grouped = df.groupby('group').first().reset_index()
    print(f'The number of extra SEP events: {len(df_grouped)}')
    df_grouped[["date", "duration"]].to_csv(
        SWEET_EVENTS_DIR / extra_sep_events_filename,
        sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/ {extra_sep_events_filename} created")


def read_extra_sweet_sep_events():
    """
    Returns a Pandas DataFrame
    with the dates where the
    detrended count rates have been above 0
    for at least three days
    Columns:
        date: Date at noon
        duration: how long the count rates were above 0
        type: SEP or Fd. Here it is always SEP.
    """
    df = pd.read_csv(SWEET_EVENTS_DIR / extra_sep_events_filename,
                     skiprows=0, sep="\t", parse_dates=['date'])
    sep_df = pd.DataFrame(df[['date', 'duration']],
                          columns=['date', 'duration'])
    sep_df['type'] = 'SEP'
    return sep_df


def find_sweet_forbush_decreases():
    """
    Find the dates where the EDAC count rate
    is 0 for more than FD_NUMBER_DAYS days
    """
    print("------ Finding SWEET Forbush decreases --------")
    resampled_df = read_resampled_df()
    print("resampled_df: ", resampled_df)
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
    detrended_df = read_detrended_rates()
    result = pd.merge(df, detrended_df, on='date', how='left')
    result.drop(columns=["group", "duration", "zero_rate"], inplace=True)
    result.to_csv(SWEET_EVENTS_DIR / zerodays_filename,
                  sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\ {zerodays_filename} created")
    # Keep only the first dates in each Forbush decrease
    df_grouped = df.groupby('group').first().reset_index()
    print(f'Number of Forbush decreases detected by SWEET: {len(df_grouped)}')
    df_grouped[["date", "duration"]].to_csv(
        SWEET_EVENTS_DIR / forbush_decrease_events_filename,
        sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\ {forbush_decrease_events_filename} \
          created")


def read_sweet_sep_events():
    """
    Returns a Pandas Dataframe with the
    start dates of each SWEET SEP event,
    and the duration
    Columns:
        start_date: Date at noon
        duration: how many days above UPPER_THRESHOLD
    """
    df = pd.read_csv(SWEET_EVENTS_DIR / sep_events_filename,
                     skiprows=0, sep="\t", parse_dates=['start_date'])
    df['type'] = 'SEP'
    # print(df['duration'].value_counts())
    return df


def read_sweet_sep_dates():
    """
    Returns a Pandas DataFrame with all
    dates that are a part of a SWEET SEP event
    Columns:
        date: Date at noon
        edac_first: First EDAC reading
        edac_last: Last EDAC reading
        daily_rate: daily rate
        gcr_component: the Savitzky-Golay fit of daily_rate
        detrended_rate: daily_rate subtracted by gcr_component
        type: SEP/FD, it is SEP here
    """
    df = pd.read_csv(SWEET_EVENTS_DIR / sep_dates_filename,
                     skiprows=0, sep="\t", parse_dates=['date'])
    df['type'] = 'SEP'

    # sep_dates = pd.DataFrame(df[['date', "detrended_rate"]],
    #                         columns=['date', "detrended_rate"])
    # sep_dates['type'] = 'SEP'
    # print(sep_dates)
    return df


def combine_sweet_sep_event_detection_methods():
    """
    Returns a Pandas Dataframe
    with all SWEET SEP events from both detection
    methods
    """
    df1 = read_extra_sweet_sep_events()
    df2 = read_sweet_sep_events()
    df = pd.concat([df1[['date', 'type']], df2[['date', 'type']]])
    df.sort_values(by='date', inplace=True)
    return df


def read_sweet_forbush_decreases():
    """
    Returns a Pandas Dataframe with
    the first dates of all
    Forbush decreases found by SWEET
    Columns:
        date: Date at noon
        duration: How many consecutive days with 0 rate
        type: SEP/Fd. Here it is Fd
    """
    df = pd.read_csv(SWEET_EVENTS_DIR / forbush_decrease_events_filename,
                     skiprows=0, sep="\t", parse_dates=['date'])
    df['type'] = 'Fd'
    return df


def read_sweet_zero_days():
    """
    Returns a Pandas DataFrame
    Columns:
        date: datetime at noon
        type: SEP/Fd. It is Fd here.
    """
    df = pd.read_csv(SWEET_EVENTS_DIR / zerodays_filename,
                     skiprows=0, sep="\t", parse_dates=['date'])
    # forbush_dates = pd.DataFrame(df['date'], columns=['date'])
    # forbush_dates['type'] = 'Fd'
    df['type'] = 'Fd'
    return df


def create_stormy_days_list():
    sep_dates = read_sweet_sep_dates()
    forbush_dates = read_sweet_zero_days()
    stormy_df = pd.concat([sep_dates, forbush_dates], ignore_index=True)
    stormy_df = stormy_df.sort_values(by='date')
    stormy_df.to_csv(SWEET_EVENTS_DIR / stormy_days_filename,
                     sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{stormy_days_filename} created")


def create_sw_event_list():
    """
    Combine the SWEET SEP events
    and the SWEET Forbush decreases
    into one Pandas DataFrame
    """
    sep_events = read_sweet_sep_events()
    sep_events = sep_events.drop(columns=['mean_rate', 'max_rate'])
    sep_events.rename(columns={"start_date": "date"}, inplace=True)
    forbush_decreases = read_sweet_forbush_decreases()
    event_df = pd.concat([sep_events, forbush_decreases], ignore_index=True)
    event_df = event_df.sort_values(by="date")
    print(f'Total number of SWEET events: {len(event_df)}')
    event_df.to_csv(SWEET_EVENTS_DIR / sweet_events_filename,
                    sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}/{sweet_events_filename} created")


def read_stormy_sweet_dates():
    """
    Returns a Pandas DataFrame
    with all stormy dates found by SWEET,
    including those parts of SEP events and
    Forbush decreases.
    Columns:
        date: Datetime at noon
        edac_first: First EDAC reading of the day
        edac_last: Last EDAC reading of the day
        daily_rate: daily count rate
        gcr_component: Savitzky-Golay fit of daily_rate
        detrended_rate: daily_rate subtracted the gcr_component
        type: SEP / Fd
    """
    df = pd.read_csv(SWEET_EVENTS_DIR / stormy_days_filename,
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def read_sweet_event_dates():
    """
    Returns a Pandas DataFrame
    with all SWEET events, including
    SEP events and Forbush decreases
    Columns:
        date: Starting date at noon for each event
        duration: duration in days
        type: SEP/Fd
    """
    df = pd.read_csv(SWEET_EVENTS_DIR / sweet_events_filename,
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def find_sweet_forbush_rolling_rate():
    df = read_detrended_rates()
    troughs = df[(df['detrended_rate'] < LOWER_THRESHOLD)]
    troughs = troughs.sort_values(by='date')
    print("The number of days below the threshold of ",
          LOWER_THRESHOLD, " is: ", len(troughs))
    troughs.to_csv(SWEET_EVENTS_DIR / zerodays_filename,
                   sep='\t', index=False)  # Save to file
    print(f"File {SWEET_EVENTS_DIR}\\ {zerodays_filename} created")


def detect_sweet_events():
    if not os.path.exists(SWEET_EVENTS_DIR):
        os.makedirs(SWEET_EVENTS_DIR)
    find_sweet_sep()
    find_sweet_forbush_decreases()
    create_stormy_days_list()
    create_sw_event_list()


def detect_sweet_events_rolling_rate():
    find_sweet_forbush_rolling_rate()
    find_sweet_sep()


if __name__ == "__main__":
    if not os.path.exists(SWEET_EVENTS_DIR):
        os.makedirs(SWEET_EVENTS_DIR)
    find_sweet_sep()
    # test_sweet_sep_variable_threshold()
    # create_sw_event_list()
    # find_sweet_sep_by_consecutive_days()
    # find_sweet_sep()
    # read_second_method_sweet_sep()
    # find_sweet_forbush_decreases()
    # read_sweet_sep_events()
    # read_sweet_fds()
    # read_sweet_forbush_decreases()
    # read_sweet_sep_events()
    # read_sweet_sep_dates()
    # read_sweet_zero_days()
    # create_sw_event_list()
    # read_sweet_event_dates()
    # find_sweet_forbush_decreases()
    # create_stormy_days_list()
    # read_sweet_event_dates()
    # find_sweet_sep_by_consecutive_days()
    # detect_sweet_events()
    # detect_sweet_events_rolling_rate()
