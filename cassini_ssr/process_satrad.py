import pandas as pd

cassini_data_dir = "cassini_ssr/cassini_data/"
filename = 'SATRAD_Output_Saturn_TAB_1h.txt'
filename = 'SATRAD_Output_Saturn_TAB_5m.txt'

def read_satrad_5m():
    column_headers = ['Time', 'Range(km)', 'SSCLat', '(e-@1MeV)', '(H+@1MeV)',
                      '(H+@10MeV)', '(H+@20MeV)', '(H+@30MeV)', '(H+@40MeV)', '(H+@50MeV)',
                      '(H+@60MeV)', '(H+@70MeV)', '(H+@80MeV)', '(H+> 10MeV)', 'SCET']
    df = pd.read_csv(cassini_data_dir + filename, sep=r'\s{2,}', engine='python', 
                     skiprows=3, header=None, names=column_headers)
    df['SCET_UTC'] =  pd.to_datetime(df['SCET'], format="%Y %b %d, %H:%M:%S.%f")


    return df[['SCET_UTC', '(H+@40MeV)']]

if __name__ == "__main__":
    df= read_satrad_5m()
    print(df)