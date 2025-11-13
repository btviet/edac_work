import time

import pandas as pd
from parameters import (
    DETREND_METHOD,
    POLYORDER_SAVGOL,
    PROCESSED_DATA_DIR,
    RATE_SAVGOL,
)
from edac_work.sweet_code.process_edac.processing_edac import read_resampled_df, read_rolling_rates
from scipy.signal import savgol_filter


def savitzky_fit_gcr(rate_df):
    """
    Apply Savitzky-Golay filter
    to the EDAC count rates.
    Returns a Pandas DataFrame
    with the Savitzky-Golay fit
    """
    df = rate_df.copy()
    y_filtered = savgol_filter(df['daily_rate'],
                               RATE_SAVGOL, POLYORDER_SAVGOL)
    df['fit'] = y_filtered
    return df


def create_detrended_rates():
    """
    Subtract the Savitzky-Golay fit
    of the EDAC count rate
    from the EDAC count rate
    """
    start_time = time.time()
    print("--------- Starting the de-trending process ---------")
    if DETREND_METHOD == 'division':
        smoothing_df = read_rolling_rates(11)
        gcr_component = savitzky_fit_gcr(smoothing_df)
        rate_df = read_rolling_rates(5)
        # rate_df = read_resampled_df()
        # gcr_component = savitzky_fit_gcr(rate_df)

    else:
        # rate_df = read_rolling_rates(5)
        # smoothing_df = read_rolling_rates(11)
        # gcr_component = savitzky_fit_gcr(smoothing_df)
        rate_df = read_resampled_df()
        gcr_component = savitzky_fit_gcr(rate_df)

    first_gcr = gcr_component['date'].iloc[0]
    last_gcr = gcr_component['date'].iloc[-1]
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

    print(gcr_component['fit'].isna().sum())
    print(len(gcr_component))
    print(detrended_df['gcr_component'].isna().sum())
    # GCR COMPONENT IS LONGER THAN RATE_DF

    gcr_component['date'] = gcr_component['date'].dt.date
    # gcr_dates = gcr_component['date'].to_list()
    detrended_df['date'] = detrended_df['date'].dt.date
    # detrended_dates = detrended_df['date'].to_list()
    # c = (set(gcr_dates) - set(detrended_dates))

    if DETREND_METHOD == 'division':
        detrended_df['detrended_rate'] = \
            detrended_df['daily_rate']/detrended_df['gcr_component'] - 1

    else:  # Detrend by subtraction
        detrended_df['detrended_rate'] = \
            detrended_df['daily_rate']-detrended_df['gcr_component']
    mean_count_rate = detrended_df['detrended_rate'].mean()
    print(f'Mean detrended count rate: {mean_count_rate}')
    filename = 'detrended_edac.txt'
    print("detrended_df: ", detrended_df)
    detrended_df.to_csv(PROCESSED_DATA_DIR / filename, sep='\t', index=False)
    print(f"File {PROCESSED_DATA_DIR}/{filename} created")
    print('Time taken: ',
          f'{time.time() - start_time:.2f}', "seconds")


def read_detrended_rates():
    """
    Returns a Pandas DataFrame with the detrended count rates
    Columns:
        date: Date at noon
        edac_first: First EDAC reading for each date
        edac_last: Last EDAC reading for each date
        daily_rate: The EDAC count rate
        gcr_component: The Savitzky-Golay fit of daily_rate
        detrended_rate: dailyrate subtracted the gcr_component
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / 'detrended_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def create_standardized_rates():
    """
    Subtract the mean and divide by standard deviation
    of the detrended EDAC count rates
    """
    start_time = time.time()
    print("----------- Starting the standardization process ---------")
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


def detrend():
    create_detrended_rates()


if __name__ == "__main__":
    create_detrended_rates()
