import pandas as pd
from parameters import DATABASE_DIR


def read_sep_database_events() -> pd.DataFrame:
    """
    Reads the SEP table which
    has been converted to a .csv file
    from the database
    Parameters
    ----------
    file_path: Path

    Returns
    -------
    pd.DataFrame
        Dataframe containing SEP onset dates, and which instrument
    """

    df = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')

    return df[["onset_time", "instrument"]]


def read_sep_events_maven():
    """
    Returns a Pandas Dataframe with

    Columns:
        onset_time
    """
    df = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    maven_detections = df[df['instrument'].str.contains('MAVEN')]
    maven_detections = maven_detections.sort_values(by='onset_time')
    return maven_detections[["onset_time"]]


def read_sep_events_rad():
    df = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    rad_detections = df[df['instrument'].str.contains('RAD')]
    rad_detections.sort_values(by='onset_time', inplace=True)
    return rad_detections[["onset_time"]]


def read_forbush_decreases_database():
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df_cleaned = df[~df['source'].str.contains('Papaioannou')]

    return df_cleaned[["onset_time", "instrument"]]


def read_forbush_decreases_rad():
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    # df = df[~df['source'].str.equals('Papaioannou et al. (2019)')]
    rad_detections = df[df['instrument'].str.contains('RAD')]
    rad_detections = \
        rad_detections[rad_detections['source'].str.contains('Guo')]

    rad_detections.loc[:, 'onset_time'] = pd.to_datetime(df['onset_time'],
                                                         dayfirst=True)
    rad_detections = rad_detections.sort_values(by='onset_time')
    return rad_detections[["onset_time"]]


def read_forbush_decreases_maven():
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    df = df[~df['source'].str.contains('Papaioannou')]
    maven_detections = df[df['instrument'].str.contains('MAVEN')]
    return maven_detections[["onset_time"]]


def read_both_sep_fd_database_events():
    """
    Returns a Pandas Dataframe
    with all SEP events and Forbush decreases
    in the event database
    Columns:
        onset_time: Starting time of event
        instrument: Instrument that made the detection
        type: SEP/Fd
    """
    df_sep = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                         skiprows=0, sep=",",
                         parse_dates=["onset_time"], date_format='%d/%m/%Y')

    df_fd = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                        skiprows=0, sep=",",
                        parse_dates=["onset_time"], date_format='%d/%m/%Y')
    # df_fd = df_fd[~df_fd['source'].str.contains('Papaioannou')]
    # df_fd = df_fd[~df_fd['source'].str.equals('Papaioannou et al. (2019)')]
    df_fd = df_fd[df_fd['source'] != 'Papaioannou et al. (2019)']
    df_sep = df_sep[["onset_time", "instrument"]]
    df_sep['type'] = 'SEP'
    df_fd = df_fd[["onset_time", "instrument"]]
    df_fd['type'] = 'Fd'
    df = pd.concat([df_sep, df_fd])
    df.sort_values(by='onset_time', inplace=True)
    df.dropna(subset=['onset_time'], inplace=True)
    return df


def create_fd_table():
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df_cleaned = df[~df['source'].str.contains('Papaioannou')]

    # df['matched_date'] = df['matched_date'].astype(object).fillna('')
    for index, row in df_cleaned.iterrows():
        row_string = (
                f"{row['onset_time'].date()} & "
                f"{row['instrument']} & "
                f"{row['source']} \\\\"
            )
        print(row_string)
    print(len(df_cleaned))


def create_sep_table():
    df = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    print(df)
    print(len(df))
    for index, row in df.iterrows():
        row_string = (
                f"{row['onset_time'].date()} & "
                f"{row['instrument']} & "
                f"{row['source']} \\\\"
            )
        print(row_string)


if __name__ == "__main__":
    # df = read_sep_event_dates()
    # df = read_sep_events_rad()
    # df = read_forbush_decreases_rad()
    create_fd_table()
    # create_sep_table()
