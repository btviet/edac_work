import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from detect_sw_events import (read_sweet_event_dates, 
                              read_sweet_sep_events,
                              read_sweet_forbush_decreases)
from parameters import SWEET_VALIDATION_DIR
from read_from_database import read_both_sep_fd_database_events


def generate_search_area(start_date):
    """
    Time period where matches in literature
    are searched for"""
    return [start_date.date() + timedelta(days=i) for i in range(-3, 4)]


def generate_search_area_for_fd(start_date):
    """
    Time period where matches in literature
    are searched for"""
    return [start_date.date() + timedelta(days=i) for i in range(-5, 2)]



def cross_check_sweet_sep_events():
    """
    Finds the SWEET event dates, and compares the
    dates found in the database with them.
    The search area is 3 days before and 3 days after
    the SWEET event date.
    When a match is found, the instrument that detected
    the event is also displayed.
    """
    sweet_df = read_sweet_sep_events()
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-07-30", "%Y-%m-%d")
    sweet_df = sweet_df[(sweet_df["start_date"] >= start_date) &
                        (sweet_df["start_date"] <= end_date)]
    
    
    database_df = read_both_sep_fd_database_events()
    database_df = database_df[(database_df["onset_time"] >= start_date) &
                              (database_df["onset_time"] <= end_date)]
    
    db_counts = database_df['type'].value_counts()

    sweet_df['date'] = sweet_df['start_date'].dt.date
    sweet_df['date'] = pd.to_datetime(sweet_df['date'])
    sweet_df['type'] = 'SEP'
    database_df['onset_time'] = database_df['onset_time'].dt.date
    sweet_dict = dict.fromkeys(sweet_df['date'], [])
    for event_date in sweet_dict.keys():
        # For each SWEET event, generate time period to look for matches
        sweet_dict[event_date] = generate_search_area(event_date)
    results = {}
    
    for key, values in sweet_dict.items():
        # For each SWEET event, check if any databased dates are in the
        # vicinity
        match_found = any(date in database_df['onset_time'].values
                          for date in values)
        results[key] = match_found
    
    # Store the results of match found/not found in df
    df = pd.DataFrame(list(results.items()),
                      columns=['date', 'match_found'])
    # The dates that found a match in the database
    dates_matched = df[df['match_found']]['date'].tolist()
    

    # dates_matched = df[df['match_found'] == True]['date'].tolist()
    # The dates that were not found in the database
    dates_not_matched = df.loc[~df['match_found'], 'date'].tolist()
    # dates_not_matched = df[df['match_found'] == False]['date'].tolist()
    database_dates = database_df['onset_time'].tolist()
    matched_list = []
    not_matched_list = []
    
    for date in dates_matched:
        search_area = generate_search_area(date)
        common_date = [date for date in search_area
                       if date in database_dates][0]
        database_df['onset_time'] = pd.to_datetime(database_df['onset_time'])
        common_date = pd.to_datetime(common_date)
        sweet_type = sweet_df[sweet_df['date'] == date]['type'].values[0]
        instrument = \
            database_df[database_df['onset_time'] ==
                        common_date]['instrument'].values[0]
        instrument_type = \
            database_df[database_df['onset_time'] ==
                        common_date]['type'].values[0]
        matched_list.append([date, sweet_type,
                             common_date, instrument, instrument_type])
    
    for date in dates_not_matched:
        sweet_type = sweet_df[sweet_df['date'] == date]['type'].values[0]
        not_matched_list.append([date, sweet_type, np.nan, np.nan, np.nan])

    df1 = pd.DataFrame(matched_list,
                       columns=['sweet_date', 'sweet_type',
                                'matched_date', 'instrument',
                                'instrument_type'])
    df2 = pd.DataFrame(not_matched_list,
                       columns=['sweet_date', 'sweet_type',
                                'matched_date', 'instrument',
                                'instrument_type'])
    df = pd.concat([df1, df2])
    df.sort_values(by='sweet_date', inplace=True)

    if not os.path.exists(SWEET_VALIDATION_DIR):
        os.makedirs(SWEET_VALIDATION_DIR)

    df.to_csv(
        SWEET_VALIDATION_DIR / "sweet_sep_validation.txt",
        sep="\t",
        index=False,
    )  # Save selected EDAc rate to file
    print(f'{len(dates_matched)} out of {len(sweet_df)}')
    #print("Dates matched: ", dates_matched)
    df_matches_only = df[df["matched_date"].notna()]


def cross_check_sweet_fds():

    sweet_df = read_sweet_forbush_decreases()
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-07-30", "%Y-%m-%d")
    print(sweet_df)
    sweet_df = sweet_df[(sweet_df["date"] >= start_date) &
                        (sweet_df["date"] <= end_date)]
    
    
    database_df = read_both_sep_fd_database_events()
    database_df = database_df[(database_df["onset_time"] >= start_date) &
                              (database_df["onset_time"] <= end_date)]
    
    db_counts = database_df['type'].value_counts()

    sweet_df['date'] = sweet_df['date'].dt.date
    sweet_df['date'] = pd.to_datetime(sweet_df['date'])
    sweet_df['type'] = 'Fd'
    database_df['onset_time'] = database_df['onset_time'].dt.date
    sweet_dict = dict.fromkeys(sweet_df['date'], [])
    for event_date in sweet_dict.keys():
        # For each SWEET event, generate time period to look for matches
        sweet_dict[event_date] = generate_search_area_for_fd(event_date)
    results = {}
    
    for key, values in sweet_dict.items():
        # For each SWEET event, check if any databased dates are in the
        # vicinity
        match_found = any(date in database_df['onset_time'].values
                          for date in values)
        results[key] = match_found
    
    # Store the results of match found/not found in df
    df = pd.DataFrame(list(results.items()),
                      columns=['date', 'match_found'])
    # The dates that found a match in the database
    dates_matched = df[df['match_found']]['date'].tolist()
    

    # dates_matched = df[df['match_found'] == True]['date'].tolist()
    # The dates that were not found in the database
    dates_not_matched = df.loc[~df['match_found'], 'date'].tolist()
    # dates_not_matched = df[df['match_found'] == False]['date'].tolist()
    database_dates = database_df['onset_time'].tolist()
    matched_list = []
    not_matched_list = []
    
    for date in dates_matched:
        search_area = generate_search_area_for_fd(date)
        common_date = [date for date in search_area
                       if date in database_dates][0]
        database_df['onset_time'] = pd.to_datetime(database_df['onset_time'])
        common_date = pd.to_datetime(common_date)
        sweet_type = sweet_df[sweet_df['date'] == date]['type'].values[0]
        instrument = \
            database_df[database_df['onset_time'] ==
                        common_date]['instrument'].values[0]
        instrument_type = \
            database_df[database_df['onset_time'] ==
                        common_date]['type'].values[0]
        matched_list.append([date, sweet_type,
                             common_date, instrument, instrument_type])
    
    for date in dates_not_matched:
        sweet_type = sweet_df[sweet_df['date'] == date]['type'].values[0]
        not_matched_list.append([date, sweet_type, np.nan, np.nan, np.nan])

    df1 = pd.DataFrame(matched_list,
                       columns=['sweet_date', 'sweet_type',
                                'matched_date', 'instrument',
                                'instrument_type'])
    df2 = pd.DataFrame(not_matched_list,
                       columns=['sweet_date', 'sweet_type',
                                'matched_date', 'instrument',
                                'instrument_type'])
    df = pd.concat([df1, df2])
    df.sort_values(by='sweet_date', inplace=True)

    if not os.path.exists(SWEET_VALIDATION_DIR):
        os.makedirs(SWEET_VALIDATION_DIR)

    df.to_csv(
        SWEET_VALIDATION_DIR / "sweet_fd_validation.txt",
        sep="\t",
        index=False,
    )  # Save selected EDAc rate to file
    print(f'{len(dates_matched)} out of {len(sweet_df)}')
    print("Dates matched: ", dates_matched)

def cross_check_sweet_old():
    """
    Finds the SWEET event dates, and compares the
    dates found in the database with them.
    The search area is 3 days before and 3 days after
    the SWEET event date.
    When a match is found, the instrument that detected
    the event is also displayed.
    """
    # sweet_df = read_sweet_sep_events()
    sweet_df = read_sweet_event_dates()
    # start_date = datetime.strptime("2014-11-01", "%Y-%m-%d") MSL/RAD
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-04-10", "%Y-%m-%d")
    sweet_df = sweet_df[(sweet_df["date"] >= start_date) &
                        (sweet_df["date"] <= end_date)]
    sweet_counts = sweet_df['type'].value_counts()
    database_df = read_both_sep_fd_database_events()
    database_df = database_df[(database_df["onset_time"] >= start_date) &
                              (database_df["onset_time"] <= end_date)]
    db_counts = database_df['type'].value_counts()

    print(f'SWEET event counts: {sweet_counts}')
    print(f'Database entries counts {db_counts}')
    sweet_df['date'] = sweet_df['date'].dt.date
    sweet_df['date'] = pd.to_datetime(sweet_df['date'])
    database_df['onset_time'] = database_df['onset_time'].dt.date
    sweet_dict = dict.fromkeys(sweet_df['date'], [])

    for event_date in sweet_dict.keys():
        # For each SWEET event, generate time period to look for matches
        sweet_dict[event_date] = generate_search_area(event_date)
    results = {}
    for key, values in sweet_dict.items():
        # For each SWEET event, check if any databased dates are in the
        # vicinity
        match_found = any(date in database_df['onset_time'].values
                          for date in values)
        results[key] = match_found

    # Store the results of match found/not found in df
    df = pd.DataFrame(list(results.items()),
                      columns=['date', 'match_found'])
    # The dates that found a match in the database
    dates_matched = df[df['match_found']]['date'].tolist()
    # dates_matched = df[df['match_found'] == True]['date'].tolist()
    # The dates that were not found in the database
    dates_not_matched = df.loc[~df['match_found'], 'date'].tolist()
    # dates_not_matched = df[df['match_found'] == False]['date'].tolist()
    database_dates = database_df['onset_time'].tolist()
    matched_list = []
    not_matched_list = []

    for date in dates_matched:
        search_area = generate_search_area(date)
        common_date = [date for date in search_area
                       if date in database_dates][0]
        database_df['onset_time'] = pd.to_datetime(database_df['onset_time'])
        common_date = pd.to_datetime(common_date)
        sweet_type = sweet_df[sweet_df['date'] == date]['type'].values[0]
        instrument = \
            database_df[database_df['onset_time'] ==
                        common_date]['instrument'].values[0]
        instrument_type = \
            database_df[database_df['onset_time'] ==
                        common_date]['type'].values[0]
        matched_list.append([date, sweet_type,
                             common_date, instrument, instrument_type])

    for date in dates_not_matched:
        sweet_type = sweet_df[sweet_df['date'] == date]['type'].values[0]
        not_matched_list.append([date, sweet_type, np.nan, np.nan, np.nan])

    df1 = pd.DataFrame(matched_list,
                       columns=['sweet_date', 'sweet_type',
                                'matched_date', 'instrument',
                                'instrument_type'])
    df2 = pd.DataFrame(not_matched_list,
                       columns=['sweet_date', 'sweet_type',
                                'matched_date', 'instrument',
                                'instrument_type'])
    df = pd.concat([df1, df2])
    df.sort_values(by='sweet_date', inplace=True)

    if not os.path.exists(SWEET_VALIDATION_DIR):
        os.makedirs(SWEET_VALIDATION_DIR)

    df.to_csv(
        SWEET_VALIDATION_DIR / "sweet_validation.txt",
        sep="\t",
        index=False,
    )  # Save selected EDAc rate to file
    print(f'{len(dates_matched)} out of {len(sweet_df)}')
    print("Dates matched: ", dates_matched)
    df_matches_only = df[df["matched_date"].notna()]

    print("df_matches_only: ", df_matches_only)

    df_matches_only['instrument'].replace('MSL/RAD', 'RAD',
                                          inplace=True)
    df_matches_only['instrument'].replace('MAVEN/SEP', 'M/SEP',
                                          inplace=True)
    df_matches_only['instrument'].replace('MSL/RAD, MAVEN/SEP',
                                          'RAD, M/SEP', inplace=True)
    """
    print("matches only: ")
    for index, row in df_matches_only.iterrows():
        #print(index)
        row_string = (
                f"{row['sweet_date'].date()} & "
                f"{row['sweet_type']} & "
                f"{row['instrument']} & "
                f"{row['matched_date'].date()} & "
                f"{row['instrument_type']} \\\\"
            )
        print(row_string)
    """
    print("all sweet_events: ")
    print(df.columns)
    df['instrument'] = df['instrument'].fillna('')
    df['instrument_type'] = df['instrument_type'].fillna('')

    # df['matched_date'] = df['matched_date'].astype(object).fillna('')
    for index, row in df.iterrows():
        row_string = (
                f"{row['sweet_date'].date()} & "
                f"{row['sweet_type']} & "
                f"{row['instrument']} & "
                f"{row['matched_date'].date()} & "
                f"{row['instrument_type']} \\\\"
            )
        print(row_string)




def create_validation_table():
    sep_df = pd.read_csv(SWEET_VALIDATION_DIR / "sweet_sep_validation.txt",
                     skiprows=0, sep="\t",
                     parse_dates=["sweet_date"])
    sep_df = sep_df.fillna("")
    #print(sep_df)

    fd_df = pd.read_csv(SWEET_VALIDATION_DIR / "sweet_fd_validation.txt",
                     skiprows=0, sep="\t",
                     parse_dates=["sweet_date"])
    
    fd_df = fd_df.fillna("")


    combined_df = pd.concat([sep_df, fd_df], ignore_index=True)

    combined_df = combined_df.sort_values(by="sweet_date")
    print(combined_df)


    for index, row in combined_df.iterrows():
            
            row_string = (
                    f"{row['sweet_date'].date()} & "
                    f"{row['sweet_type']} & "
                    f"{row['instrument']} & "
                    f"{row['matched_date']} & "
                    f"{row['instrument_type'] } \\\\"
                )
            print(row_string)

if __name__ == "__main__":
    #cross_check_sweet()
    create_validation_table()
    # cross_check_sweet_sep_events()
    #cross_check_sweet_fds()