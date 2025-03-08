import os
from datetime import timedelta

import pandas as pd
from detect_sw_events import read_sweet_sep_dates, read_stormy_sweet_dates
from detrend_edac import read_detrended_rates
from parameters import SEP_VALIDATION_DIR
from read_from_database import (read_sep_database_events, 
                                read_sep_events_rad, 
                                read_forbush_decreases_database)
from validate_forbush_decreases import read_fd_validation_results


def generate_SEP_days_old(start_date):
    # Time period where SWEET SEP is expected
    return [start_date.date() + timedelta(days=i) for i in range(-3, 4)]


# Look for SWEET SEP events 3 days before and 3 days after, 
# and SWEET FDs 1 day before and 5 days after.

def detect_verified_sep_old():
    def generate_SWEET_SEP_days(start_date):
    # Time period where SWEET SEP is expected
        return [start_date.date() + timedelta(days=i) for i in range(-3, 4)]

    def generate_SWEET_Fd_days(start_date):
    # Time period where SWEET Fd is expected
        return [start_date.date() + timedelta(days=i) for i in range(-1, 6)]

    sweet_df = read_stormy_sweet_dates()
    print(sweet_df)
    sweet_df['date'] = sweet_df['date'].dt.date
    sep_df = read_sep_database_events()  # Database SEP events
    sep_dict = dict.fromkeys(sep_df['onset_time'], [])

    for event_date in sep_dict.keys():
        sep_dict[event_date] = generate_SWEET_SEP_days(event_date)
    results = {}
    for key, values in sep_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area 0-3 days after
        match_found = any(date in sweet_df['date'].values
                          for date in values)
        if match_found:
            print(match_found)
            current_type = sweet_df[sweet_df['date'] == key]['type']
            print(current_type)
        else: 
            current_type = ''
        instrument_found = sep_df[sep_df["onset_time"]
                                  == key]["instrument"].iloc[0]
        results[key] = [match_found, current_type, instrument_found]

    for event_date in sep_dict.keys():
        sep_dict[event_date] = generate_SEP_days(event_date)

    rows = [(key, val[0], val[1], val[2]) for key, val in results.items()]

    result_df = pd.DataFrame(rows, columns=['date', 'SEP_found', 'SWEET type', 'instrument'])
    # result_df = pd.DataFrame(list(results.items()),
    # columns=['date', 'SEP_found'])
    detrended_df = read_detrended_rates()

    detrended_df['date'] = detrended_df['date'].dt.date
    detrended_df['date'] = pd.to_datetime(detrended_df['date'])
    df = pd.merge(result_df, detrended_df, how='left', on='date')
    if not os.path.exists(SEP_VALIDATION_DIR):
        os.makedirs(SEP_VALIDATION_DIR)

    df.to_csv(SEP_VALIDATION_DIR / "validation_events_sep.txt",
              sep='\t', index=False)
    print(f"File created: {SEP_VALIDATION_DIR}/validation_events_sep.txt")
    
    print(df['SEP_found'].value_counts())


def validate_verified_sep_with_sweet():
    def generate_SWEET_SEP_days(start_date):
    # Time period where SWEET SEP is expected
        return [start_date.date() + timedelta(days=i) for i in range(-3, 4)]

    def generate_SWEET_Fd_days(start_date):
    # Time period where SWEET Fd is expected
        return [start_date.date() + timedelta(days=i) for i in range(-1, 6)]

    sweet_df = read_stormy_sweet_dates()
    sweet_sep_df = sweet_df[sweet_df['type']=='SEP']
    sweet_fd_df = sweet_df[sweet_df['type']=='Fd']
    
    sweet_df['date'] = sweet_df['date'].dt.date
    sweet_sep_df['date'] = sweet_sep_df['date'].dt.date
    sweet_fd_df['date'] = sweet_fd_df['date'].dt.date
    print("sweet_fd_df:", sweet_fd_df)
    verified_sep_df = read_sep_database_events()  # Database SEP events
    #sep_dict = dict.fromkeys(sep_df['onset_time'], [])
    verified_sep_dict = dict.fromkeys(verified_sep_df['onset_time'], [])

    for event_date in verified_sep_dict.keys():
       verified_sep_dict[event_date] = generate_SWEET_SEP_days(event_date)
    results = {}
    for key, values in verified_sep_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area -3-3 days
        sweet_sep_found = any(date in sweet_sep_df['date'].values
                          for date in values)
        if sweet_sep_found:
            current_type = 'SEP'
        else: 
            current_type = ''
        instrument_found = verified_sep_df[verified_sep_df["onset_time"]
                                  == key]["instrument"].iloc[0]
        results[key] = [sweet_sep_found, current_type, instrument_found]
    rows_sep = [(key, val[0], val[1], val[2]) for key, val in results.items()]

    # Do this for SWEET Fd
    verified_sep_dict = dict.fromkeys(verified_sep_df['onset_time'], [])
    for event_date in verified_sep_dict.keys():
       verified_sep_dict[event_date] = generate_SWEET_Fd_days(event_date)
    results = {}
    for key, values in verified_sep_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area -3-3 days
        sweet_fd_found = any(date in sweet_fd_df['date'].values
                          for date in values)
        if sweet_fd_found:
            current_type = 'Fd'
        else: 
            current_type = ''
        instrument_found = verified_sep_df[verified_sep_df["onset_time"]
                                  == key]["instrument"].iloc[0]
        results[key] = [sweet_fd_found, current_type, instrument_found]

    rows_fd = [(key, val[0], val[1], val[2]) for key, val in results.items()]

    result_fd_df = pd.DataFrame(rows_fd, columns=['date', 'Fd_found', 'SWEET type', 'instrument'])
    fd_df = result_fd_df[['date', 'Fd_found', 'instrument']]
    result_sep_df = pd.DataFrame(rows_sep, columns=['date', 'SEP_found', 'SWEET type', 'instrument'])
    sep_df = result_sep_df[['date','SEP_found']]
    result_df = pd.merge(sep_df, fd_df, how='left', on='date')
    result_df["SWEET_found"] = result_df["SEP_found"] | result_df["Fd_found"]


    detrended_df = read_detrended_rates()

    detrended_df['date'] = detrended_df['date'].dt.date
    detrended_df['date'] = pd.to_datetime(detrended_df['date'])
    df = pd.merge(result_df, detrended_df, how='left', on='date')
    if not os.path.exists(SEP_VALIDATION_DIR):
        os.makedirs(SEP_VALIDATION_DIR)

    df.to_csv(SEP_VALIDATION_DIR / "validate_verified_sep.txt",
              sep='\t', index=False)
    print(f"File created: {SEP_VALIDATION_DIR}/validate_verified_sep.txt")
    
    print(df['SWEET_found'].value_counts())


def validate_verified_fd_with_sweet():
    def generate_SWEET_SEP_days(start_date):
    # Time period where SWEET SEP is expected
        return [start_date.date() + timedelta(days=i) for i in range(-5, 2)]

    def generate_SWEET_Fd_days(start_date):
    # Time period where SWEET Fd is expected
        return [start_date.date() + timedelta(days=i) for i in range(-2, 4)]

    sweet_df = read_stormy_sweet_dates()
    sweet_sep_df = sweet_df[sweet_df['type']=='SEP']
    sweet_fd_df = sweet_df[sweet_df['type']=='Fd']
    
    sweet_df['date'] = sweet_df['date'].dt.date
    sweet_sep_df['date'] = sweet_sep_df['date'].dt.date
    sweet_fd_df['date'] = sweet_fd_df['date'].dt.date

    verified_fd_df = read_forbush_decreases_database()  # Database Fd events

    verified_fd_dict = dict.fromkeys(verified_fd_df['onset_time'], [])

    # Check with SWEET SEPs
    for event_date in verified_fd_dict.keys():
       verified_fd_dict[event_date] = generate_SWEET_SEP_days(event_date)
    results = {}
    for key, values in verified_fd_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area -3-3 days
        sweet_sep_found = any(date in sweet_sep_df['date'].values
                          for date in values)
        if sweet_sep_found:
            current_type = 'SEP'
        else: 
            current_type = ''
        instrument_found = verified_fd_df[verified_fd_df["onset_time"]
                                  == key]["instrument"].iloc[0]
        results[key] = [sweet_sep_found, current_type, instrument_found]
    rows_sep = [(key, val[0], val[1], val[2]) for key, val in results.items()]

    # Do this for SWEET Fd
    verified_fd_dict = dict.fromkeys(verified_fd_df['onset_time'], [])
    for event_date in verified_fd_dict.keys():
       verified_fd_dict[event_date] = generate_SWEET_Fd_days(event_date)
    results = {}
    for key, values in verified_fd_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area -3-3 days
        sweet_fd_found = any(date in sweet_fd_df['date'].values
                          for date in values)
        if sweet_fd_found:
            current_type = 'Fd'
        else: 
            current_type = ''
        instrument_found = verified_fd_df[verified_fd_df["onset_time"]
                                  == key]["instrument"].iloc[0]
        results[key] = [sweet_fd_found, current_type, instrument_found]

    rows_fd = [(key, val[0], val[1], val[2]) for key, val in results.items()]

    result_fd_df = pd.DataFrame(rows_fd, columns=['date', 'Fd_found', 'SWEET type', 'instrument'])
    fd_df = result_fd_df[['date', 'Fd_found', 'instrument']]
    result_sep_df = pd.DataFrame(rows_sep, columns=['date', 'SEP_found', 'SWEET type', 'instrument'])
    sep_df = result_sep_df[['date','SEP_found']]
    result_df = pd.merge(sep_df, fd_df, how='left', on='date')
    result_df["SWEET_found"] = result_df["SEP_found"] | result_df["Fd_found"]

    detrended_df = read_detrended_rates()

    detrended_df['date'] = detrended_df['date'].dt.date
    detrended_df['date'] = pd.to_datetime(detrended_df['date'])
    df = pd.merge(result_df, detrended_df, how='left', on='date')

    df.to_csv(SEP_VALIDATION_DIR / "validate_verified_fd.txt",
              sep='\t', index=False)
    print(f"File created: {SEP_VALIDATION_DIR}/validate_verified_fd.txt")
    
    print(df['SWEET_found'].value_counts())


def detect_real_sep():
    sweet_sep_df = read_sweet_sep_dates()  # SWEET SEP dates
    sweet_sep_df['date'] = sweet_sep_df['date'].dt.date
    sep_df = read_sep_database_events()  # Database SEP events
    sep_dict = dict.fromkeys(sep_df['onset_time'], [])

    for event_date in sep_dict.keys():
        sep_dict[event_date] = generate_SEP_days(event_date)
    results = {}
    for key, values in sep_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area 0-3 days after
        match_found = any(date in sweet_sep_df['date'].values
                          for date in values)
        instrument_found = sep_df[sep_df["onset_time"]
                                  == key]["instrument"].iloc[0]
        results[key] = [match_found, instrument_found]

    for event_date in sep_dict.keys():
        sep_dict[event_date] = generate_SEP_days(event_date)

    rows = [(key, val[0], val[1]) for key, val in results.items()]

    result_df = pd.DataFrame(rows, columns=['date', 'SEP_found', 'instrument'])
    # result_df = pd.DataFrame(list(results.items()),
    # columns=['date', 'SEP_found'])
    detrended_df = read_detrended_rates()

    detrended_df['date'] = detrended_df['date'].dt.date
    detrended_df['date'] = pd.to_datetime(detrended_df['date'])
    df = pd.merge(result_df, detrended_df, how='left', on='date')
    if not os.path.exists(SEP_VALIDATION_DIR):
        os.makedirs(SEP_VALIDATION_DIR)

    df.to_csv(SEP_VALIDATION_DIR / "validation_events_sep.txt",
              sep='\t', index=False)
    print(f"File created: {SEP_VALIDATION_DIR}/validation_events_sep.txt")
    
    print(df['SEP_found'].value_counts())


def validate_msl_rad_dates_sep():
    """

    """
    sweet_df = read_sweet_sep_dates()
    sweet_df['date'] = sweet_df['date'].dt.date
    rad_df = read_sep_events_rad()
    sep_dict = dict.fromkeys(rad_df['onset_time'], [])
    for event_date in sep_dict.keys():
        sep_dict[event_date] = generate_next_3_days(event_date)
    results = {}
    for key, values in sep_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area -3 -3 days after
        match_found = any(date in sweet_df['date'].values for date in values)
        results[key] = match_found
    df = pd.DataFrame(list(results.items()),
                      columns=['onset_time', 'SEP_found'])
    file_name = "validate_msl_rad_sep.txt"
    if not os.path.exists(SEP_VALIDATION_DIR):
        os.makedirs(SEP_VALIDATION_DIR)
    df.to_csv(SEP_VALIDATION_DIR / file_name,
              sep='\t', index=False)
    print(f"File created: {SEP_VALIDATION_DIR}/{file_name}")


def read_sep_validation_results():
    return pd.read_csv(SEP_VALIDATION_DIR / 'validation_events_sep.txt',
                       skiprows=0, sep="\t", parse_dates=['date'])


def analyze_sep_validation_results():
    df = read_sep_validation_results()
    print(df)
    number_events = len(df)
    sweet_detected = df['SEP_found'].sum()
    print(f"Number of events: {number_events} \
           SWEET found: {sweet_detected}")


def create_literature_events_found_table_specialization():
    """
    Uses results from validate_forbush_decreases.py
    """
    sep_df = read_sep_validation_results()
    sep_df = sep_df[["date", "SEP_found", "instrument"]]
    sep_df["type"] = 'SEP'
    fd_df = read_fd_validation_results()
    fd_df = fd_df[["date", "FD_found", "instrument"]]
    fd_df["type"] = 'FD'
    sep_df = sep_df[sep_df["SEP_found"]]
    fd_df = fd_df[fd_df["FD_found"]]
    # sep_df = sep_df[sep_df["SEP_found"]==True]
    # fd_df = fd_df[fd_df["FD_found"]==True]
    df = pd.concat([sep_df[["date", "instrument", "type"]],
                    fd_df[["date", "instrument", "type"]]])
    df.sort_values(by="date", inplace=True)

    # fd_df = read_fd_validation_results()
    # print(fd_df)
    for index, row in df.iterrows():
        row_string = (
                f"{row['date'].date()} & "
                f"{row['instrument']} & "
                f"{row['type']} \\\\"
            )
        print(row_string)


def validate_sep_onsets():
    detect_real_sep()
    analyze_sep_validation_results()


if __name__ == "__main__":
    # detect_real_sep()
    #create_literature_events_found_table()
    # detect_real_sep()
    # detect_verified_sep()
    # validate_verified_sep_with_sweet()
    validate_verified_fd_with_sweet()
    # analyze_sep_validation_results()
    # validate_msl_rad_dates_sep()
