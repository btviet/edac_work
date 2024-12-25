from parameters import MEX_ASPERA_DIR 
import pandas as pd


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


def resample_mex_ima_bg_counts():
    df = clean_up_mex_ima_bg_counts()
    df.set_index('datetime', inplace=True)
    daily_max = df.resample('D')['bg_counts'].max()

    result = df.loc[df.groupby(df.index.date)['bg_counts'].idxmax()]
    result.reset_index(inplace=True)

    return result
   
if __name__ == "__main__":
    # create_ima_background_counts_df()
    #df = read_mex_ima_bg_counts().sort_values(by="bg_counts", ascending=False)
    #print(df.iloc[0:10])
    df = resample_mex_ima_bg_counts().sort_values(by="bg_counts", ascending=False)
    print(df.iloc[0:20])
    #df = clean_up_mex_ima_bg_counts().sort_values(by="bg_counts", ascending=False)
    #print(df.iloc[40:60])
