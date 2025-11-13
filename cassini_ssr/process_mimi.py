import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

cassini_data_dir = "cassini_ssr/cassini_data/"


def process_mimi_data_2004_2016():
    column_headers = ['endwindow','year','doy','hour','cnt','tL','X','Y','Z','tP8']
    df = pd.read_csv(cassini_data_dir + "MIMI_2004_2016_P8orbit5min.txt")
    df.columns = column_headers

    df['endwindow_utc'] = pd.to_datetime(df['year'].astype(str), format='%Y') + \
                 pd.to_timedelta(df['doy'] - 1, unit='D') + \
                 pd.to_timedelta(df['hour'], unit='h')
    df['startwindow_utc'] = df['endwindow_utc'].shift(1)

    df = df[~df['tP8'].str.contains('NaN', na=False)]
    df = df.iloc[1:]
    return df[['startwindow_utc', 'endwindow_utc', 'cnt', 'tL', 'X', 'Y', 'Z', 'tP8']]

def process_mimi_data_2017_2017():
    column_headers = ['endwindow','year','doy','hour','cnt','tL','X','Y','Z','tP8']
    df = pd.read_csv(cassini_data_dir + "MIMI_2017_2017_P8orbit5min.txt")
    df.columns = column_headers
    df['endwindow_utc'] = pd.to_datetime(df['year'].astype(str), format='%Y') + \
                 pd.to_timedelta(df['doy'] - 1, unit='D') + \
                 pd.to_timedelta(df['hour'], unit='h')
    df['startwindow_utc'] = df['endwindow_utc'].shift(1)
    #df = df[~df['tP8'].str.contains('NaN', na=False)]

    df = df.iloc[1:]
    return df[['startwindow_utc', 'endwindow_utc', 'cnt', 'tL', 'X', 'Y', 'Z', 'tP8']]

def create_merged_mimi():
    df_one = process_mimi_data_2004_2016()
    df_two = process_mimi_data_2017_2017()
    df = pd.concat([df_one, df_two], axis=0)

    df.to_csv(
        cassini_data_dir + "cassini_mimi_merged.csv",
        sep="\t",
        index=False,
    )  

def read_cassini_mimi():
    df = pd.read_csv(cassini_data_dir + 'cassini_mimi_merged.csv',
        sep='\t', parse_dates=['endwindow_utc'])
    return df


def resample_cassini_mimi():
    df = read_cassini_mimi()
    df['tP8'] = df['tP8'].astype(float)
    df.set_index('datetime', inplace=True)
    df_hourly = df.resample('h').mean().dropna(how='all').reset_index()

    df_hourly.to_csv(
        cassini_data_dir + "resampled_cassini_mimi_merged.csv",
        sep="\t",
        index=False,
    )  

def read_resampled_cassini_mimi():
    return pd.read_csv(cassini_data_dir + 'resampled_cassini_mimi_merged.csv',
    sep='\t', parse_dates=['datetime'])

if __name__ == "__main__":
    # process_mimi_data_2004_2016()
    #create_merged_mimi()
    #resample_cassini_mimi()
    # resample_cassini_mimi()
    df = read_cassini_mimi()
    #process_mimi_data_2017_2017()
    #print(df.dtypes)
    #print(df.sort_values(by='tP8'))
    # print(df)