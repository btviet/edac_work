from datetime import timedelta

import pandas as pd
from detect_sw_events import read_forbush_sweet_dates, read_sep_sweet_dates
from parameters import LOCAL_DIR


def read_cme_events() -> pd.DataFrame:
    """ Reads the CME table which
    has been converted to a .csv file
    from the database

    Parameters
    ----------
    file_path: Path


    Returns
    -------
    pd.DataFrame
        Dataframe containing CME eruption dates
    """

    df = pd.read_csv(LOCAL_DIR / 'cme_events.csv',
                     skiprows=0, sep=",",
                     parse_dates=["eruption_date"], date_format='%d-%b-%y')
    return df[["eruption_date"]]


def generate_next_7_days(start_date):
    # Time period where SEP is expected after CME eruption
    return [start_date + timedelta(days=i) for i in range(0, 8)]


def generate_fd_search_area(start_date):
    # Time period where FD is expected after CME eruption
    return [start_date + timedelta(days=i) for i in range(3, 9)]


def detect_real_sep():
    sep_df = read_sep_sweet_dates()
    sep_df['date'] = sep_df['date'].dt.date
    cme_df = read_cme_events()
    cme_df["eruption_date"] = cme_df["eruption_date"].dt.date
    cme_dict = dict.fromkeys(cme_df['eruption_date'], [])
    for event_date in cme_dict.keys():
        cme_dict[event_date] = generate_next_7_days(event_date)
    results = {}
    for key, values in cme_dict.items():
        # For each CME eruption,
        # look for SWEET SEPs in the time area 7 days after
        match_found = any(date in sep_df['date'].values for date in values)
        results[key] = match_found
    df = pd.DataFrame(list(results.items()),
                      columns=['eruption_date', 'SEP_found'])
    df.to_csv(LOCAL_DIR / "validation_events_sep.txt", sep='\t', index=False)
    print(f"File created: {LOCAL_DIR}/validation_events_sep.txt")


def detect_real_fd():
    fd_df = read_forbush_sweet_dates()
    fd_df['date'] = fd_df['date'].dt.date

    cme_df = read_cme_events()
    cme_df["eruption_date"] = cme_df["eruption_date"].dt.date
    cme_dict = dict.fromkeys(cme_df['eruption_date'], [])

    for event_date in cme_dict.keys():
        cme_dict[event_date] = generate_fd_search_area(event_date)

    results = {}
    for key, values in cme_dict.items():
        match_found = any(date in fd_df['date'].values for date in values)
        results[key] = match_found
    df = pd.DataFrame(list(results.items()),
                      columns=['eruption_date', 'FD_found'])
    df.to_csv(LOCAL_DIR / "validation_events_fd.txt", sep='\t', index=False)
    print(f"File created: {LOCAL_DIR}/validation_events_fd.txt")


def combine_findings_sep_fd():
    sep_df = pd.read_csv(LOCAL_DIR / 'validation_events_sep.txt',
                         skiprows=0, sep="\t", parse_dates=['eruption_date'])
    fd_df = pd.read_csv(LOCAL_DIR / 'validation_events_fd.txt',
                        skiprows=0, sep="\t", parse_dates=['eruption_date'])
    merged_df = pd.merge(sep_df, fd_df, on='eruption_date')
    merged_df['result'] = merged_df[['SEP_found', 'FD_found']].any(axis=1)
    merged_df.to_csv(LOCAL_DIR / "validation_events_combined.txt",
                     sep='\t', index=False)
    print(f"File created: {LOCAL_DIR}/validation_events_combined.txt")


def read_validation_results():
    return pd.read_csv(LOCAL_DIR / 'validation_events_combined.txt',
                       skiprows=0, sep="\t", parse_dates=['eruption_date'])


def analyze_validation_results():
    df = read_validation_results()
    number_events = len(df)
    sweet_detected = df['result'].sum()
    print(f"Number of events: {number_events} \
           SWEET found: {sweet_detected}")


def validate_cme_eruptions():
    detect_real_sep()
    detect_real_fd()
    combine_findings_sep_fd()
