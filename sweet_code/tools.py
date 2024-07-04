import pandas as pd


def print_missing_dates(date_column):
    # example of date_column input: df['datetime].dt.date
    # remove the time from datetime object
    start_date = date_column.iloc[0]
    end_date = date_column.iloc[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    missing_dates = date_range[~date_range.isin(date_column)]
    print("Start date is ", start_date, ". End date is ", end_date)
    print("Missing dates: ", missing_dates)
    print(len(missing_dates))
