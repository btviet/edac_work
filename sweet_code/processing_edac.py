import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from file_paths import get_data_dir

load_dotenv()


def read_rawedac(file_path):
    # Reads the patched MEX EDAC
    df = pd.read_csv(file_path / "patched_mex_edac.txt",
                     skiprows=0, sep="\t", parse_dates=['datetime'])
    return df


def read_zero_set_correct(file_path):
    df = pd.read_csv(
        file_path / 'zerosetcorrected_edac.txt',
        skiprows=0, sep="\t", parse_dates=['datetime'])
    return df


def read_resampled_df(file_path):
    # Retrieves the resampled and zeroset corrected edac
    df = pd.read_csv(file_path / 'resampled_corrected_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def create_zero_set_correct(file_path):
    # Create the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    print("--------- Starting the zeroset correction ---------")
    df = read_rawedac(file_path)
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
    df.to_csv(file_path / 'zerosetcorrected_edac.txt', sep='\t', index=False)
    print("File  ", "zerosetcorrected_edac.txt created")
    print('Time taken to perform zero-set correction and create files: ',
          f'{time.time() - start_time:.2f}', "seconds")


def create_resampled_edac(file_path):
    """
     Resamples the zero set corrected EDAC counter
     to have one reading each day,
     and save the resulting dataframe to a textfile
     """
    start_time = time.time()
    print('Starting the resampling to a daily frequency process')
    zerosetcorrected_df = read_zero_set_correct(file_path)
    zerosetcorrected_df = zerosetcorrected_df.set_index('datetime')

    last_df = zerosetcorrected_df.resample('D').last().ffill()
    df_resampled = zerosetcorrected_df.resample('D').first().ffill()
    df_resampled.reset_index(inplace=True)
    last_df.reset_index(inplace=True)
    df_resampled['edac_last'] = last_df['edac']
    df_resampled.rename(columns={'datetime': 'date', 'edac': 'edac_first'},
                        inplace=True)
    df_resampled['daily_rate'] = df_resampled['edac_last']\
        - df_resampled['edac_first']
    # set datetime of each date to noon
    df_resampled['date'] = df_resampled['date']+pd.Timedelta(hours=12)
    filename = 'resampled_corrected_edac.txt'
    df_resampled.to_csv(file_path / filename, sep='\t', index=False)
    print('File ', filename, ' created')
    print('Time taken: ',
          f'{time.time() - start_time:.2f}', "seconds")


def process_raw_edac():
    file_path = get_data_dir()
    create_zero_set_correct(file_path)
    create_resampled_edac(file_path)
