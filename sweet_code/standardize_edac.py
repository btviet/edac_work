import pandas as pd
from file_paths import get_data_dir
from parameters import RATE_SAVGOL
from processing_edac import read_resampled_df
from scipy.signal import savgol_filter


def savitzky_fit_gcr(file_path):
    """Retrieves the last

    Parameters
    ----------
    timestamp : pd.Timestamp
        Timestamp that will be queried.

    Returns
    -------
    pd.DataFrame
        Dataframe containing all ais positions as rows.
    """

    rate_df = read_resampled_df(file_path)
    savgolwindow = RATE_SAVGOL
    polyorder = 2
    # Apply filtering to the EDAC rates with large spikes removed
    y_filtered = savgol_filter(rate_df['daily_rate'], savgolwindow, polyorder)
    rate_df['fit'] = y_filtered
    return rate_df


def create_detrended_rates(file_path):
    gcr_component = savitzky_fit_gcr(file_path)
    first_gcr = gcr_component['date'].iloc[0]
    last_gcr = gcr_component['date'].iloc[-1]
    rate_df = read_resampled_df(file_path)
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
    detrended_df.to_csv(file_path / filename, sep='\t', index=False)
    print("File ", filename, " created")


def read_detrended_rates(file_path):
    df = pd.read_csv(file_path / 'detrended_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def create_standardized_rates(file_path):
    detrended_df = read_detrended_rates(file_path)
    detrended_mean = detrended_df['detrended_rate'].mean()
    detrended_std = detrended_df['detrended_rate'].std()
    detrended_df['standardized_rate'] = (detrended_df['detrended_rate'] -
                                         detrended_mean)/detrended_std
    filename = 'standardized_edac.txt'
    detrended_df.to_csv(file_path / filename,
                        sep='\t', index=False)
    print("File ", filename, " created")


def read_standardized_rates(file_path):
    df = pd.read_csv(file_path / 'standardized_edac.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def standardize():
    file_path = get_data_dir()
    create_detrended_rates(file_path)
    create_standardized_rates(file_path)
