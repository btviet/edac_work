import time

import pandas as pd
from parameters import PROCESSED_DATA_DIR, RATE_SAVGOL
from processing_edac import read_resampled_df
from scipy.signal import savgol_filter


def savitzky_fit_gcr():
    """
    Apply Savitzky-Golay filter
    to the EDAC count rates
    """
    rate_df = read_resampled_df()
    savgolwindow = RATE_SAVGOL
    polyorder = 2
    y_filtered = savgol_filter(rate_df['daily_rate'], savgolwindow, polyorder)
    rate_df['fit'] = y_filtered
    return rate_df


def create_detrended_rates():
    """
    Subtract the Savitzky-Golay fit
    of the EDAC count rate
    from the EDAC count rate
    """
    start_time = time.time()
    print("Starting the de-trending process")
    gcr_component = savitzky_fit_gcr()
    first_gcr = gcr_component['date'].iloc[0]
    last_gcr = gcr_component['date'].iloc[-1]
    rate_df = read_resampled_df()
    first_rate = rate_df['date'].iloc[0]
    last_rate = rate_df['date'].iloc[-1]
    if first_gcr >= first_rate:
        rate_df = rate_df[rate_df['date'] >= first_gcr]
    else:
        gcr_component = gcr_component[gcr_component['date'] >= first_rate]

    if last_gcr >= last_rate:
        gcr_component = gcr_component[gcr_component['date'] <= last_rate]

    else:
        rate_df = rate_df[rate_df['date'] <= last_gcr]

    rate_df.reset_index(drop=True, inplace=True)
    gcr_component.reset_index(drop=True, inplace=True)

    detrended_df = rate_df.copy()

    detrended_df['gcr_component'] = gcr_component['fit']
    # Detrending by subtraction
    detrended_df['detrended_rate'] = \
        detrended_df['daily_rate']-detrended_df['gcr_component']

    filename = 'detrended_edac.txt'
    detrended_df.to_csv(PROCESSED_DATA_DIR / filename, sep='\t', index=False)
    print(f"File {PROCESSED_DATA_DIR}/{filename} created")
    print('Time taken: ',
          f'{time.time() - start_time:.2f}', "seconds")


def read_detrended_rates():
    df = pd.read_csv(PROCESSED_DATA_DIR / 'detrended_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def create_standardized_rates():
    """
    Subtract the mean and divide by standard deviation
    of the detrended EDAC count rates
    """
    start_time = time.time()
    print("Starting the standardization process")
    detrended_df = read_detrended_rates()
    detrended_mean = detrended_df['detrended_rate'].mean()
    detrended_std = detrended_df['detrended_rate'].std()
    detrended_df['standardized_rate'] = (detrended_df['detrended_rate'] -
                                         detrended_mean)/detrended_std
    filename = 'standardized_edac.txt'
    detrended_df.to_csv(PROCESSED_DATA_DIR / filename,
                        sep='\t', index=False)
    print(f"File {PROCESSED_DATA_DIR}/{filename} created")
    print('Time taken: ',
          f'{time.time() - start_time:.2f}', "seconds")


def read_standardized_rates():
    df = pd.read_csv(PROCESSED_DATA_DIR / 'standardized_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def standardize():
    create_detrended_rates()
    create_standardized_rates()
