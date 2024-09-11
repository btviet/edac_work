import pandas as pd
from parameters import DATABASE_DIR


def read_sep_event_dates() -> pd.DataFrame:
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
        Dataframe containing CME eruption dates
    """

    df = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')

    return df[["onset_time"]]


def read_sep_events_maven():
    df = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    maven_detections = df[df['instrument'].str.contains('MAVEN')]
    return maven_detections[["onset_time"]]


def read_sep_events_rad():
    df = pd.read_csv(DATABASE_DIR / 'sep_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    rad_detections = df[df['instrument'].str.contains('RAD')]
    return rad_detections[["onset_time"]]


def read_forbush_decreases():
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    return df[["onset_time"]]


def read_forbush_decreases_rad():
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    rad_detections = df[df['instrument'].str.contains('RAD')]
    return rad_detections[["onset_time"]]


def read_forbush_decreases_maven():
    df = pd.read_csv(DATABASE_DIR / 'forbush_database.csv',
                     skiprows=0, sep=",",
                     parse_dates=["onset_time"], date_format='%d/%m/%Y')
    df.dropna(subset=['instrument'], inplace=True)  # Drop NaN values
    maven_detections = df[df['instrument'].str.contains('MAVEN')]
    return maven_detections[["onset_time"]]


if __name__ == "__main__":
    df = read_sep_event_dates()
    df = read_sep_events_rad()
    df = read_forbush_decreases_rad()
    print(df)
