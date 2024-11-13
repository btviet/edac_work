import os
from datetime import timedelta

import pandas as pd
from detect_sw_events import read_sweet_zero_days
from detrend_edac import read_detrended_rates
from parameters import DATABASE_DIR, FORBUSH_VALIDATION_DIR
from read_from_database import (
    read_forbush_decreases_database,
    read_forbush_decreases_rad,
)

msl_rad_fd_filename = "validate_msl_rad_fd.txt"
sweet_fd_filename = "validation_events_fd.txt"


def generate_search_dates(start_date):
    # Time period where SEP is expected SEP onset
    start_date = pd.to_datetime(start_date, dayfirst=True)
    return [start_date.date() + timedelta(days=i) for i in range(-3, 4)]


def detect_real_fd():
    sweet_fd_df = read_sweet_zero_days()
    # print(sweet_fd_df)
    sweet_fd_df['date'] = sweet_fd_df['date'].dt.date
    fd_database = read_forbush_decreases_database()
    print(fd_database)
    fd_dict = dict.fromkeys(fd_database['onset_time'], [])
    for event_date in fd_dict.keys():
        fd_dict[event_date] = generate_search_dates(event_date)
    results = {}
    for key, values in fd_dict.items():
        match_found = any(date in sweet_fd_df['date'].values
                          for date in values)
        results[key] = match_found
        instrument_found = fd_database[fd_database["onset_time"]
                                       == key]["instrument"].iloc[0]
        results[key] = [match_found, instrument_found]
    rows = [(key, val[0], val[1]) for key, val in results.items()]

    result_df = pd.DataFrame(rows, columns=['date', 'FD_found', 'instrument'])

    # result_df = pd.DataFrame(list(results.items()),
    # columns=['date', 'FD_found'])
    detrended_df = read_detrended_rates()

    detrended_df['date'] = detrended_df['date'].dt.date
    detrended_df['date'] = pd.to_datetime(detrended_df['date'])
    df = pd.merge(result_df, detrended_df, how='left', on='date')
    if not os.path.exists(FORBUSH_VALIDATION_DIR):
        os.makedirs(FORBUSH_VALIDATION_DIR)

    df.to_csv(FORBUSH_VALIDATION_DIR / sweet_fd_filename,
              sep='\t', index=False)
    print(f"File created: {FORBUSH_VALIDATION_DIR}/{sweet_fd_filename}")


def read_fd_validation_results():
    return pd.read_csv(FORBUSH_VALIDATION_DIR / sweet_fd_filename,
                       skiprows=0, sep="\t", parse_dates=['date'])


def analyze_fd_validation_results():
    df = read_fd_validation_results()
    number_events = len(df)
    sweet_detected = df['FD_found'].sum()
    print(f"Number of events: {number_events} \
           SWEET found: {sweet_detected}")


def validate_msl_rad_dates_fd():
    # Contains all days included in a Forbush decrease
    sweet_df = read_sweet_zero_days()
    # sweet_df = read_stormy_sweet_dates()
    sweet_df['date'] = sweet_df['date'].dt.date
    rad_df = read_forbush_decreases_rad()
    print("rad_df: ", rad_df)
    rad_dict = dict.fromkeys(rad_df['onset_time'], [])
    for event_date in rad_dict.keys():
        rad_dict[event_date] = generate_search_dates(event_date)
    results = {}
    for key, values in rad_dict.items():
        match_found = any(date in sweet_df['date'].values for date in values)
        results[key] = match_found
    df = pd.DataFrame(list(results.items()),
                      columns=['onset_time', 'Fd_found'])
    df['onset_time'] = pd.to_datetime(df['onset_time'])
    df_with_instrument = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                                     skiprows=0, sep=",",
                                     parse_dates=["onset_time"],
                                     date_format='%d/%m/%Y')
    df_with_instrument['onset_time'] = df_with_instrument['onset_time'].dt.date
    df_with_instrument['onset_time'] = \
        pd.to_datetime(df_with_instrument['onset_time'])

    result = pd.merge(df, df_with_instrument, on='onset_time', how='left')

    if not os.path.exists(FORBUSH_VALIDATION_DIR):
        os.makedirs(FORBUSH_VALIDATION_DIR)
    result.to_csv(FORBUSH_VALIDATION_DIR / msl_rad_fd_filename,
                  sep='\t', index=False)

    print(f"File created: {FORBUSH_VALIDATION_DIR} /{msl_rad_fd_filename}")
    print(f"Total number of MSL/RAD Forbush decreases: {len(rad_df)}")
    print(f'Number of FDs found: {df["Fd_found"].sum()}')
    maven_count = result['instrument'].str.contains('MAVEN').sum()
    print(f"Number of FDs also found by MAVEN: {maven_count}")
    print(result)
    for index, row in result.iterrows():
        # print(index)
        if "MAVEN/SEP" in row["instrument"]:
            row["instrument"] = "Yes"
        else:
            row["instrument"] = "No"
        if row["Fd_found"]:
            # if row["Fd_found"] == True:
            row["Fd_found"] = "Yes"
        else:
            row["Fd_found"] = "No"
        row_string = (
                f"{row['onset_time'].date()} & "
                f"{row['instrument']} & "
                f"{row['Fd_found']} \\\\"
                # f"{row['matched_date'].date()} & "
                # f"{row['instrument_type']} \\\\"
            )
        print(row_string)


def read_msl_rad_fd_validation_result():
    df = pd.read_csv(FORBUSH_VALIDATION_DIR / msl_rad_fd_filename,
                     skiprows=0, sep="\t", parse_dates=['onset_time'])
    return df.sort_values(by='onset_time')


if __name__ == "__main__":
    # validate_msl_rad_dates_fd()
    # detect_real_fd()
    # analyze_fd_validation_results()
    detect_real_fd()
    analyze_fd_validation_results()
    # validate_msl_rad_dates_fd()
