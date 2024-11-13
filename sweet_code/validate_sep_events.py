import os
from datetime import timedelta

import pandas as pd
from detect_sw_events import read_sweet_sep_dates
from detrend_edac import read_detrended_rates
from parameters import SEP_VALIDATION_DIR
from read_from_database import read_sep_database_events, read_sep_events_rad
from validate_forbush_decreases import read_fd_validation_results


def generate_next_3_days(start_date):
    # Time period where SWEET EP is expected
    return [start_date.date() + timedelta(days=i) for i in range(-3, 4)]


def detect_real_sep():
    sweet_sep_df = read_sweet_sep_dates()  # SWEET SEP dates
    sweet_sep_df['date'] = sweet_sep_df['date'].dt.date
    sep_df = read_sep_database_events()  # Database SEP events
    sep_dict = dict.fromkeys(sep_df['onset_time'], [])

    for event_date in sep_dict.keys():
        sep_dict[event_date] = generate_next_3_days(event_date)
    results = {}
    for key, values in sep_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area 0-3 days after
        match_found = any(date in sweet_sep_df['date'].values
                          for date in values)
        instrument_found = sep_df[sep_df["onset_time"]
                                  == key]["instrument"].iloc[0]
        results[key] = [match_found, instrument_found]

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


def create_literature_events_found_table():
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
    create_literature_events_found_table()
    # analyze_sep_validation_results()
    # validate_msl_rad_dates_sep()
