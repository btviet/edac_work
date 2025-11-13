from parameters import MSL_RAD_DIR
from datetime import datetime
import pandas as pd

def process_msl_rad_data():
    """

    """
    header_list = ['sol', 'date', 'time', 'B_dose', 'B_dose_err', 'E_dose', 'E_dose_err']
    df = pd.read_csv(MSL_RAD_DIR / 'RAD_data_sols_1_4415.csv',
                     delim_whitespace=True,
                     names=header_list,
                     skiprows=1)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df[['sol', 'datetime', 'B_dose', 'B_dose_err', 'E_dose', 'E_dose_err']]
    df.to_csv(MSL_RAD_DIR / 'msl_rad_doses.txt', sep='\t', index=False)
 
def read_msl_rad_doses():
    df = pd.read_csv(MSL_RAD_DIR / 'msl_rad_doses.txt',
                     sep = '\t',
                     parse_dates=['datetime'])
    df = df[
        df["datetime"] < datetime.strptime("2024-07-31", "%Y-%m-%d")
    ]
    return df


def process_filtered_rad_e_doses():
    header_list = ['date', 'time', 'year', 'doy','sol', 'E_dose', 'E_dose_filtered']
    df = pd.read_csv(MSL_RAD_DIR / 'doseE_sol_filter_update.txt',
                        delim_whitespace=True,
                        names=header_list,
                        skiprows=1)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df[['datetime', 'E_dose', 'E_dose_filtered']]
    df.to_csv(MSL_RAD_DIR / 'msl_rad_filtered_E_doses.txt', sep='\t', index=False)
    
def read_msl_rad_filtered_e_doses():
    df = pd.read_csv(MSL_RAD_DIR / 'msl_rad_filtered_E_doses.txt',
                     sep = '\t',
                     parse_dates=['datetime'])
    return df

if __name__ == "__main__":
    #df = read_msl_rad_doses()
    #print(df)
    df = read_msl_rad_filtered_e_doses()
    print(df)