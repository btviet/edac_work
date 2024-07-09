import pandas as pd
from parameters import OLD_SWEET_DIR
from validate_events import (
    generate_fd_search_area,
    generate_next_7_days,
    read_cme_events,
)


def read_spikes():
    """
    Read the peaks and troughs from old SWEET
    """
    sep_df = pd.read_csv(OLD_SWEET_DIR / 'spikesabove_SW11.txt',
                         skiprows=0, sep="\t", parse_dates=['date'])
    fd_df = pd.read_csv(OLD_SWEET_DIR / 'spikesbelow_SW11.txt',
                        skiprows=0, sep="\t", parse_dates=['date'])
    return sep_df, fd_df


def read_sep_sweet_dates_old():
    return pd.read_csv(OLD_SWEET_DIR / 'spikesabove_SW11.txt',
                       skiprows=0, sep="\t", parse_dates=['date'])


def read_fd_sweet_dates_old():
    return pd.read_csv(OLD_SWEET_DIR / 'spikesbelow_SW11.txt',
                       skiprows=0, sep="\t", parse_dates=['date'])


def detect_real_sep():
    sep_df = read_sep_sweet_dates_old()
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
    df.to_csv(OLD_SWEET_DIR / "validation_events_sep.txt",
              sep='\t', index=False)
    print(f"File created: {OLD_SWEET_DIR}/validation_events_sep.txt")


def detect_real_fd():
    fd_df = read_fd_sweet_dates_old()
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
    df.to_csv(OLD_SWEET_DIR / "validation_events_fd.txt",
              sep='\t', index=False)
    print(f"File created: {OLD_SWEET_DIR}/validation_events_fd.txt")


def combine_findings_sep_fd():
    sep_df = pd.read_csv(OLD_SWEET_DIR / 'validation_events_sep.txt',
                         skiprows=0, sep="\t",
                         parse_dates=['eruption_date'])
    fd_df = pd.read_csv(OLD_SWEET_DIR / 'validation_events_fd.txt',
                        skiprows=0, sep="\t",
                        parse_dates=['eruption_date'])
    merged_df = pd.merge(sep_df, fd_df, on='eruption_date')
    merged_df['result'] = \
        merged_df[['SEP_found', 'FD_found']].any(axis=1)
    merged_df.to_csv(OLD_SWEET_DIR / "validation_events_combined.txt",
                     sep='\t', index=False)
    print(f"File created:\
          {OLD_SWEET_DIR}/validation_events_combined.txt")


def read_validation_results_old():
    return pd.read_csv(OLD_SWEET_DIR / 'validation_events_combined.txt',
                       skiprows=0, sep="\t", parse_dates=['eruption_date'])


def analyze_validation_results():

    df = read_validation_results_old()
    number_events = len(df)
    sweet_detected = df['result'].sum()
    print(f"Number of events: {number_events} \
           SWEET found: {sweet_detected}")


combine_findings_sep_fd()
analyze_validation_results()
