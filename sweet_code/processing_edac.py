import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from parameters import PROCESSED_DATA_DIR, RAW_DATA_DIR

load_dotenv()


def read_rawedac():
    # Reads the patched MEX EDAC
    df = pd.read_csv(RAW_DATA_DIR / "patched_mex_edac.txt",
                     skiprows=0, sep="\t", parse_dates=['datetime'])
    return df


def read_zero_set_correct():
    df = pd.read_csv(
        PROCESSED_DATA_DIR / 'zerosetcorrected_edac.txt',
        skiprows=0, sep="\t", parse_dates=['datetime'])
    return df


def read_resampled_df():
    # Retrieves the resampled and zeroset corrected edac
    df = pd.read_csv(PROCESSED_DATA_DIR / 'resampled_corrected_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def create_zero_set_correct():
    # Create the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    print("--------- Starting the zeroset correction ---------")
    df = read_rawedac()
    diffs = df.edac.diff()
    # Finding the indices where the EDAC counter decreases
    indices = np.where(diffs < 0)[0]
    print("This EDAC data set was zero-set ", len(indices), " times.")
    for i in range(0, len(indices)):
        prev_value = df.loc[[indices[i]-1]].values[-1][-1]
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
     and save the resulting dataframe to a textfile
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

    # Set datetime of each date to noon
    df_resampled['date'] = df_resampled['date']+pd.Timedelta(hours=12)
    filename = 'resampled_corrected_edac.txt'
    df_resampled.to_csv(PROCESSED_DATA_DIR / filename, sep='\t', index=False)
    print(f"File {PROCESSED_DATA_DIR}/{filename} created")
    print('Time taken: ',
          f'{time.time() - start_time:.2f}', "seconds")


def process_raw_edac():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    create_zero_set_correct()
    create_resampled_edac()


if __name__ == "__main__":
    # create_zero_set_correct()
    create_resampled_edac()
