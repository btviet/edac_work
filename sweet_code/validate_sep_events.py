from datetime import timedelta

import pandas as pd
from detect_sw_events import read_sep_sweet_dates
from parameters import LOCAL_DIR, SEP_VALIDATION_DIR


def read_sep_events() -> pd.DataFrame:
    """ Reads the SEP table which
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

    df = pd.read_csv(LOCAL_DIR / 'sep_events.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    return df[["onset_time"]]


def generate_next_3_days(start_date):
    # Time period where SEP is expected SEP onset
    return [start_date.date() + timedelta(days=i) for i in range(0, 3)]


def detect_real_sep():
    sweet_df = read_sep_sweet_dates()
    sweet_df['date'] = sweet_df['date'].dt.date
    print(sweet_df, "sweet_df")
    sep_df = read_sep_events()
    sep_dict = dict.fromkeys(sep_df['onset_time'], [])
    for event_date in sep_dict.keys():
        sep_dict[event_date] = generate_next_3_days(event_date)
    print(sep_dict, "sep_dict")
    results = {}
    for key, values in sep_dict.items():
        # For each SEP onset time
        # look for SWEET SEPs in the time area 0-3 days after
        match_found = any(date in sweet_df['date'].values for date in values)
        results[key] = match_found
    df = pd.DataFrame(list(results.items()),
                      columns=['onset_time', 'SEP_found'])
    df.to_csv(SEP_VALIDATION_DIR / "validation_events_sep.txt",
              sep='\t', index=False)
    print(f"File created: {SEP_VALIDATION_DIR}/validation_events_sep.txt")


def read_sep_validation_results():
    return pd.read_csv(SEP_VALIDATION_DIR / 'validation_events_sep.txt',
                       skiprows=0, sep="\t", parse_dates=['onset_time'])


def analyze_sep_validation_results():
    df = read_sep_validation_results()
    number_events = len(df)
    sweet_detected = df['SEP_found'].sum()
    print(f"Number of events: {number_events} \
           SWEET found: {sweet_detected}")


def validate_sep_onsets():
    # detect_real_sep()
    analyze_sep_validation_results()
