import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from parameters import PROCESSED_DATA_DIR, RAW_DATA_DIR

load_dotenv()


def read_rawedac():
    """
    Returns a Pandas DataFrame with the
    patched MEX EDAC.
    Columns:
        datetime: datetime of the reading
        edac: sampled value

    """
    df = pd.read_csv(RAW_DATA_DIR / "patched_mex_edac.txt",
                     skiprows=0, sep="\t", parse_dates=['datetime'])
    return df


def read_zero_set_correct():
    """
    Returns a Pandas Dataframe with
    the zeroset-corrected EDAC
    Columns:
        datetime: datetime of the reading
        edac: zeroset-corrected EDAC value
    """
    df = pd.read_csv(
        PROCESSED_DATA_DIR / 'zerosetcorrected_edac.txt',
        skiprows=0, sep="\t", parse_dates=['datetime'])
    return df


def read_resampled_df():
    """
    Returns a Pandas DataFrame
    with the the resampled and zeroset corrected edac
    Columns:
        date: date set at noon every day
        edac_first: the value of EDAC counter
        edac_last: the last value of the EDAC counter for that day
        daily_rate: the difference between the current
                and the previous consecutive edac_last
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / 'resampled_corrected_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def create_zero_set_correct():
    """
    Create the zero-set corrected dataframe of the raw EDAC counter
    """
    start_time = time.time()
    print("--------- Starting the zeroset correction ---------")
    df = read_rawedac()
    diffs = df.edac.diff()
    # Finding the indices where the EDAC counter decreases
    indices = np.where(diffs < 0)[0]
    print("This EDAC data set was zero-set ", len(indices), " times.")
    for i in range(0, len(indices)):
        prev_value = df.loc[[indices[i]-1]].values[-1][-1]
        print("prev_value: ", prev_value)
        print(df.loc[[indices[i]-1]])
        if i == len(indices)-1:  # The last time the EDAC counter goes to zero
            df.loc[indices[i]:, 'edac'] = \
                df.loc[indices[i]:, 'edac'] + prev_value
        else:
            df.loc[indices[i]:indices[i+1]-1, 'edac'] = \
                df.loc[indices[i]:indices[i+1]-1, 'edac'] + prev_value
    filename = "zerosetcorrected_edac.txt"
    df.to_csv(PROCESSED_DATA_DIR / filename, sep='\t', index=False)
    print(f"File {PROCESSED_DATA_DIR}/{filename} created")
    print('Time taken to perform zero-set correction and create files: ',
          f'{time.time() - start_time:.2f}', "seconds")


def create_resampled_edac():
    """
     Resamples the zero set corrected EDAC counter
     to have one reading each day,
     and save the resulting dataframe to a textfile.
     The daily rate is calculated by taking
     the difference between two consecutive
     last readings.
     """
    start_time = time.time()
    print('---- Starting the resampling to a daily frequency process ----')
    zerosetcorrected_df = read_zero_set_correct()
    zerosetcorrected_df = zerosetcorrected_df.set_index('datetime')

    last_df = zerosetcorrected_df.resample('D').last()
    df_resampled = zerosetcorrected_df.resample('D').first()
    df_resampled.reset_index(inplace=True)
    last_df.reset_index(inplace=True)
    df_resampled['edac_last'] = last_df['edac']
    df_resampled.rename(columns={'datetime': 'date', 'edac': 'edac_first'},
                        inplace=True)
    df_resampled['daily_rate'] = np.nan  # Initialize daily rate column

    # Treat gaps
    nan_indices = df_resampled.loc[df_resampled["edac_first"].isna()].index
    nan_sequences = []
    group = [nan_indices[0]]  # Start the first group with the first element

    for i in range(1, len(nan_indices)):
        if nan_indices[i] - nan_indices[i - 1] == 1:
            # Check if the current number is consecutive
            group.append(nan_indices[i])
        else:
            nan_sequences.append(group)
            group = [nan_indices[i]]
    nan_sequences.append(group)

    for sequence in nan_sequences:
        """
        For each gap, calculate the mean between the last
        reading before the gap and the first reading after gap.
        Populate the rates of the dates in the gap with this mean.
        """
        last_reading = df_resampled.iloc[sequence[0]-1]["edac_last"]
        next_reading = df_resampled.iloc[sequence[-1]+1]["edac_first"]

        mean = (next_reading-last_reading)/len(sequence)

        df_resampled.loc[sequence, 'daily_rate'] = mean
        # First date after a NaN sequence gets a daily rate
        # by subtracting the last reading and the first reading
        # for the day
        df_resampled.at[sequence[-1] + 1, 'daily_rate'] = \
            (
            df_resampled.at[sequence[-1] + 1, 'edac_last'] -
            df_resampled.at[sequence[-1] + 1, 'edac_first']
            )

    # For all remaining rows, the daily rate is the difference
    # between the last reading and the previous last reading
    df_resampled.loc[df_resampled['daily_rate'].isna(), 'daily_rate'] = \
        df_resampled['edac_last'].diff()

    # Set the count rate for the first date in the data set
    df_resampled.at[0, 'daily_rate'] = \
        (
        df_resampled.at[0, 'edac_last'] -
        df_resampled.at[0, 'edac_first'])

    # Set datetime of each date to noon
    df_resampled['date'] = df_resampled['date']+pd.Timedelta(hours=12)
    filename = 'resampled_corrected_edac.txt'
    df_resampled.to_csv(PROCESSED_DATA_DIR / filename, sep='\t', index=False)
    print(f"File {PROCESSED_DATA_DIR}/{filename} created")
    print('Time taken: ',
          f'{time.time() - start_time:.2f}', "seconds")


def calculate_rolling_window_rate(days_window):

    start_time = time.time()
    print("--------- Calculating the daily rates ---------")
    # Fetch resampled EDAC data
    # read output from create_resampled_corrected_edac()
    df_resampled = read_resampled_df()
    df_resampled = df_resampled[["date", "edac_last"]]
    df_resampled.rename(columns={"edac_last": "edac"}, inplace=True)
    # The starting date in the data
    startdate = df_resampled['date'][df_resampled.index[days_window//2]].date()
    # The last date and time in the dataset
    lastdate = df_resampled['date'][df_resampled.index[-days_window//2]].date()
    print("The starting date is ", startdate, "\nThe last date is ", lastdate)
    df_resampled['startwindow_edac'] = \
        df_resampled['edac'].shift(days_window//2)
    df_resampled['startwindow_date'] = \
        df_resampled['date'].shift(days_window//2)
    df_resampled['endwindow_date'] = \
        df_resampled['date'].shift(-(days_window//2))
    df_resampled['endwindow_edac'] = \
        df_resampled['edac'].shift(-(days_window//2))

    df_resampled['edac_diff'] = df_resampled['endwindow_edac'] - \
        df_resampled['startwindow_edac']
    df_resampled['daily_rate'] = df_resampled['edac_diff'] / days_window
    # Remove all columns except for the date and the daily rate
    new_df = df_resampled[['date', 'edac', 'daily_rate']]
    # Remove rows without a daily rate
    new_df = new_df[new_df['daily_rate'].notna()]

    filename = str(days_window) + '_daily_rate.txt'
    new_df.to_csv(PROCESSED_DATA_DIR / filename, sep='\t', index=False)
    print("File  ", str(days_window) + "_daily_rate.txt created")
    print('Time taken to create rate_df ',
          f'{time.time() - start_time:.2f}', "seconds")


def read_rolling_rates(days_window):
    file_name = str(days_window) + "_daily_rate.txt"
    df = pd.read_csv(PROCESSED_DATA_DIR / file_name,
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def process_raw_edac():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    create_zero_set_correct()
    create_resampled_edac()


if __name__ == "__main__":
    create_zero_set_correct()
    # calculate_rolling_window_rate()
    # df = read_rawedac()
    # print(df)
    # print(df['daily_rate'].value_counts())
    #df = read_zero_set_correct()
    #print(df)
    
