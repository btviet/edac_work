from datetime import timedelta

import pandas as pd
from detect_sw_events import read_stormy_dates


def read_cme_events(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path / 'cme_events.csv',
                     skiprows=0, sep=",", parse_dates=["eruption_date"])
    return df[["eruption_date"]]


def generate_next_7_days(start_date):
    return [start_date + timedelta(days=i) for i in range(1, 8)]


def detect_cme_events():
    stormy_df = read_stormy_dates()
    stormy_df['date'] = stormy_df['date'].dt.date
    cme_df = read_cme_events()
    cme_df["eruption_date"] = cme_df["eruption_date"].dt.date
    cme_dict = dict.fromkeys(cme_df['eruption_date'], [])
    for event_date in cme_dict.keys():
        cme_dict[event_date] = generate_next_7_days(event_date)
    results = {}

    for key, values in cme_dict.items():
        # Check if any date in the list is present in the DataFrame column
        match_found = any(date in stormy_df['date'].values for date in values)
        results[key] = match_found
    for key, value in results.items():
        print(f"{key}: {value}")
