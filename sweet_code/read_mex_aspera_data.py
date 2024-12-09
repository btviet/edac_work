from parameters import MEX_ASPERA_DIR 
import pandas as pd


ima_bg_counts_folder = "mex_aspera_ima_background"


df = pd.read_csv(MEX_ASPERA_DIR / "IMA_20223651847MOM01.CSV",
                     skiprows=0, sep=",",
                     )
column_names = ['start_time', 'stop_time', 'data_type_name',
            'data_type_id', 'data_name', 'data_unit', 'value_1', 'value_2', 'value_3']


def create_ima_background_counts_df():
    path = MEX_ASPERA_DIR / ima_bg_counts_folder
    column_headers = ['datetime', 'value1', 'value2']
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
                     sep = '\t')
    return df

if __name__ == "__main__":
    read_mex_ima_bg_counts()