from parameters import MEX_ASPERA_DIR 
import pandas as pd
from scipy.signal import savgol_filter

ima_bg_counts_folder = "mex_aspera_ima_background"

# from http://rhea.umea.irf.se/~peje/mex/hk/bkg/
df = pd.read_csv(MEX_ASPERA_DIR / "IMA_20223651847MOM01.CSV",
                     skiprows=0, sep=",",
                     )
column_names = ['start_time', 'stop_time', 'data_type_name',
            'data_type_id', 'data_name', 'data_unit', 'value_1', 'value_2', 'value_3']


def create_ima_background_counts_df():
    path = MEX_ASPERA_DIR / ima_bg_counts_folder
    column_headers = ['datetime', 'bg_counts', 'total_counts']
    total_df = pd.DataFrame(columns=column_headers)
    for i in range(2004, 2025):
        df = pd.read_csv(path / f'E_IM{str(i)}.dat.txt',
                     names=column_headers,
                     parse_dates=['datetime'],
                     sep=" "
                     )
        total_df = pd.concat([total_df, df])
    total_df.to_csv(path / 'mex_ima_background_counts.txt', sep='\t', index=False)
    

def read_mex_ima_bg_counts():
    """
    Returns:
        Pandas DataFrame with three columns:
            datetime
            bg_counts  
            total_counts

    """
    path = MEX_ASPERA_DIR / ima_bg_counts_folder
    df = pd.read_csv(path / 'mex_ima_background_counts.txt',
                     sep = '\t',
                     parse_dates=['datetime'])
    return df



def clean_up_mex_ima_bg_counts():
    df = read_mex_ima_bg_counts()
    df = df[df["bg_counts"]<3000]
    df = df[df["bg_counts"]>=0.5]
    return df

import matplotlib.pyplot as plt

def apply_sg_filter_to_ima_bg_counts():

    df = read_mex_ima_bg_counts()

    print(df.sort_values(by="bg_counts", ascending=False))
    df = df[df["bg_counts"]>=0.5]
    df = df[df["bg_counts"]<3000]
    df["smoothed"] = savgol_filter(df['bg_counts'],
                               100, 2)
    df = df[df["smoothed"]>= 0.1]

    # plt.figure()
    #  df = df[df["smoothed"]<5]
    # plt.hist(df["smoothed"])
    # plt.show()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   figsize=(8, 5))
    ax1.plot(df['datetime'], df['bg_counts'])
    # df = df[df["bg_counts"]>=0.5]
    # df = df[df["bg_counts"]<3000]
    # ax0 = ax1.twinx()
    # ax0.plot(df['datetime'], df['bg_counts'])

    ax2.plot(df['datetime'], df['smoothed'], label="smoothed")
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylim([10e-3, 10e6])
    ax2.set_ylim([10e-6, 10e4])
    ax2.set_xlabel("Date")
    ax1.set_ylabel("Counts")
    ax2.set_ylabel("Counts")
    ax1.grid()
    ax2.grid()
    plt.show()

def resample_mex_ima_bg_counts():
    """
    Find the highest count value for each day
    """
    df = clean_up_mex_ima_bg_counts()
    df.set_index('datetime', inplace=True)
    daily_max = df.resample('D')['bg_counts'].max()

    result = df.loc[df.groupby(df.index.date)['bg_counts'].idxmax()]
    result.reset_index(inplace=True)

    return result
   


def read_aspera_sw_moments():
    column_headers = ["datetime", "density", "speed", "temperature", "flag"]
    df = pd.read_csv(MEX_ASPERA_DIR/ 'MomentsOrb.ascii.txt',
                     delim_whitespace=True,
                     names=column_headers,
                     parse_dates = ["datetime"],
                     skiprows=22)
    df = df[df["flag"]==0] 
    return df


if __name__ == "__main__":
    # create_ima_background_counts_df()
    # df = read_mex_ima_bg_counts().sort_values(by="bg_counts", ascending=False)
    # print(df.iloc[0:10])
    # df = resample_mex_ima_bg_counts().sort_values(by="bg_counts", ascending=False)
    # print(df.iloc[0:20])
    # df = clean_up_mex_ima_bg_counts().sort_values(by="bg_counts", ascending=False)
    # print(df.iloc[40:60])
    read_aspera_sw_moments()
    # apply_sg_filter_to_ima_bg_counts()
    # df = read_mex_ima_bg_counts()
    # print(df)