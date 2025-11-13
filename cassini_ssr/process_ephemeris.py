import pandas as pd

cassini_data_dir = "cassini_ssr/cassini_data/"
filename = 'Ephemeris1hr.txt'

def read_ephemeris():
    column_headers= ['year', 'doy', 'hr', 'mi', 'scdist', 'subsc_lat', 'subsc_lon']
    #df = pd.read_csv(cassini_data_dir + filename, sep=r'\s{2,}', engine='python', 
    #                 skiprows=3, header=None, names=column_headers)
    df = pd.read_csv(cassini_data_dir + filename, sep=r'\s{1,}', engine='python',
    skiprows=1, names=column_headers)
    print(df)

    df['datetime'] = pd.to_datetime(df['year'].astype(str), format='%Y') + \
                 pd.to_timedelta(df['doy'] - 1, unit='D') + \
                 pd.to_timedelta(df['hr'], unit='h') + \
                 pd.to_timedelta(df['mi'], unit='m')
    return df[['datetime', 'scdist', 'subsc_lat', 'subsc_lon']]

if __name__ == "__main__":
    df = read_ephemeris()
    print(df)