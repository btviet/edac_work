import os

import pandas as pd
from parameters import TOOLS_OUTPUT_DIR
from processing_edac import read_rawedac


def find_missing_dates():
    df = read_rawedac()
    date_column = df['datetime'].dt.date
    start_date = date_column.iloc[0]
    end_date = date_column.iloc[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    missing_dates = date_range[~date_range.isin(date_column)]
    df = pd.DataFrame(missing_dates, columns=['date'])
    print("Start date is ", start_date, ". End date is ", end_date)
    print("Missing dates: ", missing_dates)
    print(len(missing_dates))

    file_name = "missing_edac_dates.txt"

    if not os.path.exists(TOOLS_OUTPUT_DIR):
        os.makedirs(TOOLS_OUTPUT_DIR)
    df.to_csv(TOOLS_OUTPUT_DIR / file_name,
              sep='\t', index=False)  # Save to file


def read_missing_dates():
    df = pd.read_csv(TOOLS_OUTPUT_DIR / "missing_edac_dates.txt",
                     skiprows=0, sep="\t", parse_dates=['date'])
    return df


def find_sampling_frequency():
    df = read_rawedac()
    df['time_difference'] = df['datetime'].diff()
    df['time_difference_in_minutes'] = \
        df['time_difference'].dt.total_seconds() / 60
    difference_mean = df['time_difference_in_minutes'].mean()
    print(f"Mean time difference in minutes: {difference_mean}")
    number_of_days = len(set(df['datetime'].dt.date))
    print(f"Average number of samples per day: {len(df)/number_of_days}", )
    df = df.sort_values(by="time_difference_in_minutes")
    # df['time_difference_in_minutes'] =
    # np.log(df["time_difference_in_minutes"])
    grouped_df = df.groupby('time_difference_in_minutes').count()

    bins = [0, 0.5, 1, 5, 10, 60, 24*60]  # Define the interval edges
    # Label for each interval
    labels = ['0-0.5', '0.5-1', '1-5', '5-10', '10-60', '60-1440']

    df['binned'] = pd.cut(df['time_difference_in_minutes'], bins=bins,
                          labels=labels, right=False)
    # grouped_df = df.groupby('time_difference_in_minutes').count()
    grouped_df = df.groupby('binned').count()
    print("grouped_df: ", grouped_df["datetime"])
    # df = df[df["time_difference_in_minutes"]<50]

    if not os.path.exists(TOOLS_OUTPUT_DIR):
        os.makedirs(TOOLS_OUTPUT_DIR)
    df.to_csv(TOOLS_OUTPUT_DIR / "sampling_frequency.txt",
              sep="\t",
              index=False)

    # plt.figure()
    # plt.plot(df['datetime'], df['time_difference_in_minutes'])
    # plt.show()

    # plt.figure()
    # plt.hist(df['time_difference_in_minutes'], bins=100)
    # plt.xlabel('Time difference in minutes')
    # plt.show()


def edac_increments():
    """
    Check if the EDAC counter increases by more than one

    """
    df = read_rawedac()
    df['edac_increment'] = df['edac'].diff()
    df.sort_values(by="edac_increment", inplace=True)
    print(df)


if __name__ == "__main__":
    # edac_increments()
    # find_sampling_frequency()
    find_missing_dates()
    read_missing_dates()
