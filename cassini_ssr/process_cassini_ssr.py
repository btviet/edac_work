import os
from pathlib import Path

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

cassini_data_dir = "cassini_ssr/cassini_data/"





def read_ssr_with_latradius(filename):
    #filename = '20112016_SSR_withlatRadius.csv'
    # filename = '20042008_SSR_withLatRadius.csv'
    # filename = '20092010_SSR.csv'
    df = pd.read_csv(cassini_data_dir + filename)
    def increment_doy(match):
        year = match.group(1)
        doy = int(match.group(2)) + 1
        return f"{year}-{doy:03d}T"

    if filename != '20042008_SSR_withLatRadius.csv':
        df['SCET_shifted'] = df['SCET'].str.replace(r'(\d{4})-(\d{3})T', increment_doy, regex=True)
        #df = df[~df['SCET'].str.contains('-000', na=False)]
        df['SCET_UTC'] = pd.to_datetime(df['SCET_shifted'], format='%Y-%jT%H')

    else:
        df['SCET_UTC'] = pd.to_datetime(df['SCET'], format='%Y-%jT%H')
    df = df.sort_values(by='SCET_UTC')
    # df['UTC Time'] = df['Decimal'].apply(lambda x: datetime.utcfromtimestamp(x))
    df = df[['SCET_UTC', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']]
    return df


def plot_ssr_with_latradius(filename):
    df = read_ssr_with_latradius(filename)
    periapses = find_periapsis_apoapsis(filename)
    start_date = datetime.strptime("2007-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2007-12-01", "%Y-%m-%d")
    # df = df[(df["SCET_UTC"] >= start_date) & (df["SCET_UTC"] <= end_date)]
    # print(df)
    # df = df[df["SCET_UTC"] <= end_date] 
    fig, ax1 = plt.subplots(figsize=(10, 6)) 
    mean = df['SSR-B-SBE'].mean()
    # print(mean)
    ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')
    #ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    #ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    ax1.set_ylim([-1, 1000])
    for date in periapses:
        ax1.axvline(x=date, color="black", linestyle='dashed',
                    linewidth="1")
    ax1.set_title("SSR-B Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()


    df.to_csv(
        "temp_df.csv",
        sep="\t",
        index=False,
    )  

    plt.show()


def find_periapsis_apoapsis(filename):
    df = pd.read_csv(cassini_data_dir + filename)

    df = df[~df['SCET'].str.contains('-000', na=False)]
    df['SCET_UTC'] = pd.to_datetime(df['SCET'], format='%Y-%jT%H')
    df = df.sort_values(by='SCET_UTC')
    periapsis_dates = df[df['Unnamed: 7']=='PERIAPSIS']
    df = df[['SCET_UTC', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']]
    periapsis_list = periapsis_dates['SCET_UTC'].tolist()
    print(len(periapsis_list))
    return periapsis_list

################

def find_periapsis_apoapsis_v2(filename):
    column_headers = ['decimal', 'hex_time', 
                      'SCET', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE', '7',
                      '8', '9', '10', '11', '12', '13', '14', '15', '16']
    df = pd.read_csv(cassini_data_dir + filename, names=column_headers)

    df = df[~df['SCET'].str.contains('-000', na=False)]
    df['SCET_UTC'] = pd.to_datetime(df['SCET'], format='%Y-%jT%H')
    df = df.sort_values(by='SCET_UTC')
    periapsis_dates = df[df['7']=='PERIAPSIS']
    #print(df)
    #df = df[['SCET_UTC', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']]
    periapsis_list = periapsis_dates['SCET_UTC'].tolist()
    print(len(periapsis_list))
    return periapsis_list


def read_ssr_sbedbe_data(filename):
    # filename = 'SSR_SBEDBE_Data_EOM.csv'
    column_headers = ['decimal', 'hex_time', 
                      'SCET', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']
    df = pd.read_csv(cassini_data_dir + filename)
    df = df.iloc[:, :7]
    df.columns=column_headers
    df = df[~df['SCET'].str.contains('-000', na=False)]
    df['SCET_UTC'] = pd.to_datetime(df['SCET'], format='%Y-%jT%H')
    df = df.sort_values(by='SCET_UTC')

    df = df[['SCET_UTC', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']]
    return df


def plot_ssr_sbedbe(filename):
    df = read_ssr_sbedbe_data(filename)
    periapses = find_periapsis_apoapsis_v2(filename)
    fig, ax1 = plt.subplots(figsize=(10, 6)) 
    mean = df['SSR-B-SBE'].mean()

    # print(mean)
    ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')
    ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    #ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    # ax1.set_ylim([-1, 1000])
    for date in periapses:
        ax1.axvline(x=date, color="black", linestyle='dashed',
                    linewidth="1")
    # ax1.set_title("SSR-B Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()


def read_ssr_sbedbe_data01312017():
    filename = 'SSR_SBEDBE_Data01312017.csv'
    df = pd.read_csv(cassini_data_dir + filename)
    df = df.iloc[:, :7]
    column_headers = ['decimal', 'hex_time', 
                      'SCET', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']
    df.columns = column_headers
    df = df[~df['SCET'].str.contains('-000', na=False)]
    df['SCET_UTC'] = pd.to_datetime(df['SCET'], format='%Y-%jT%H')
    df = df[['SCET_UTC', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']]
    return df


def plot_ssr_sbedbe_data01312017():
    df = read_ssr_sbedbe_data01312017()
    filename = 'SSR_SBEDBE_Data01312017.csv'
    periapses = find_periapsis_apoapsis_v2(filename)
    fig, ax1 = plt.subplots(figsize=(10, 6)) 
    mean = df['SSR-B-SBE'].mean()

    # print(mean)
    #ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')
    ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    #ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    # ax1.set_ylim([-1, 1000])
    for date in periapses:
        ax1.axvline(x=date, color="black", linestyle='dashed',
                    linewidth="1")
    ax1.set_title("SSR-B Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()


def read_ssr_2004_2016():
    column_headers = ['year', 'doy',  
                      'hour', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']
    filename = 'SSR_2004_2016.csv'
    df = pd.read_csv(cassini_data_dir + filename)
    df.columns = column_headers
    df['datetime'] = pd.to_datetime(df['year'].astype(str), format='%Y') + \
                 pd.to_timedelta(df['doy'] - 1, unit='D') + \
                 pd.to_timedelta(df['hour'], unit='h')
    df = df[['datetime', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE']]
    return df


def plot_ssr_2004_2016():
    df = read_ssr_2004_2016()
    print(df)
    fig, ax1 = plt.subplots(figsize=(10, 6)) 

    ax1.plot(df['datetime'], df['SSR-A-SBE'], label='SSR-A-SBE')
    ax1.plot(df['datetime'], df['SSR-B-SBE'], label='SSR-B-SBE')

    ax1.set_title("SSR-B Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()


def plot_20092010_ssr():
    filename = '20092010_SSR.csv'
    df = read_ssr_with_latradius(filename)
    print(df)
    fig, ax1 = plt.subplots(figsize=(10, 6)) 

    #ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')
    ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')

    ax1.set_title("SSR-A Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()


def create_list_of_apses_times():

    # 2016-2017
    filename = 'SSR_SBEDBE_Data01312017.csv'
    df = pd.read_csv(cassini_data_dir + filename)
    column_headers = ['decimal', 'hex_time', 
                      'SCET', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE', 'apses']
    df.columns = column_headers

    def increment_doy(match):
        year = match.group(1)
        doy = int(match.group(2)) + 1
        return f"{year}-{doy:03d}T"

    df['SCET_shifted'] = df['SCET'].str.replace(r'(\d{4})-(\d{3})T', increment_doy, regex=True)
    df['SCET_UTC'] = pd.to_datetime(df['SCET_shifted'], format='%Y-%jT%H')
    data_2017 = df[(df['apses'] =='PERIAPSIS') | (df['apses'] == 'APOAPSIS')][['SCET_UTC', 'apses']]
    #combined_df = pd.concat([periapsis_dates, apoapsis_dates], axis=0)
    

    filename = 'SSR_SBEDBE_Data_EOM.csv'
    column_headers = ['decimal', 'hex_time', 
                      'SCET', 'SSR-A-SBE', 'SSR-A-DBE', 'SSR-B-SBE', 'SSR-B-DBE', 'apses']
    df = pd.read_csv(cassini_data_dir + filename)
    df = df.iloc[:, :8]
    df.columns=column_headers
    df['SCET_shifted'] = df['SCET'].str.replace(r'(\d{4})-(\d{3})T', increment_doy, regex=True)

    df['SCET_UTC'] = pd.to_datetime(df['SCET_shifted'], format='%Y-%jT%H')
    data_eom = df[(df['apses'] =='PERIAPSIS') | (df['apses'] == 'APOAPSIS')][['SCET_UTC', 'apses']]
    #print(data_eom)

    filename = '20042008_SSR_withLatRadius.csv'
    df = pd.read_csv(cassini_data_dir + filename)
    df = df.iloc[:, :8]
    df['SCET_shifted'] = df['SCET'].str.replace(r'(\d{4})-(\d{3})T', increment_doy, regex=True)

    df['SCET_UTC'] = pd.to_datetime(df['SCET_shifted'], format='%Y-%jT%H')
    data_eom = df[(df['apses'] =='PERIAPSIS') | (df['apses'] == 'APOAPSIS')][['SCET_UTC', 'apses']]
    print(df)


def plot_together():
    df_eom = read_ssr_sbedbe_data('SSR_SBEDBE_Data_EOM.csv')
    df_ssr_data = read_ssr_sbedbe_data01312017()
    df_with_latradius = read_ssr_with_latradius('20112016_SSR_withlatRadius.csv')
    df_with_latradius_early = read_ssr_with_latradius('20042008_SSR_withLatRadius.csv')
    df_2009 = read_ssr_with_latradius('20092010_SSR.csv')
    df_new = read_ssr_2004_2016()
    fig, ax1 = plt.subplots(figsize=(10, 6)) 

    ax1.plot(df_with_latradius_early['SCET_UTC'], df_with_latradius_early['SSR-A-SBE'], 
             label='20042008_SSR_withLatRadius',
             linestyle='dashed')
    ax1.plot(df_2009['SCET_UTC'], df_2009['SSR-A-SBE'], 
             label='20092010_SSR', linewidth=1,
             linestyle='dashed')
    ax1.plot(df_with_latradius['SCET_UTC'], df_with_latradius['SSR-A-SBE'], 
             label='20112016_SSR_withlatRadius',
             linestyle='dashed')
    ax1.plot(df_eom['SCET_UTC'], df_eom['SSR-A-SBE'], 
             label='SSR_SBEDBE_Data_EOM',
             linestyle='dashed')
    #ax1.plot(df_ssr_data['SCET_UTC'], df_ssr_data['SSR-A-SBE'], 
    #         label='SSR_SBEDBE_Data01312017',
    #         linestyle='dashed')
    ax1.plot(df_new['datetime'], df_new['SSR-A-SBE'], 
             label='SSR_2004_2016',
             linestyle='dashed',
             linewidth=0.5,
             color='black')
    
    # ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    ax1.set_title("SSR-A Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6)) 

    ax1.plot(df_with_latradius_early['SCET_UTC'], df_with_latradius_early['SSR-B-SBE'], 
             label='20042008_SSR_withLatRadius',
             linestyle='dashed')
    ax1.plot(df_2009['SCET_UTC'], df_2009['SSR-B-SBE'], 
             label='20092010_SSR',
             linestyle='dashed')
    
    ax1.plot(df_with_latradius['SCET_UTC'], df_with_latradius['SSR-B-SBE'], 
             label='20112016_SSR_withlatRadius',
             linestyle='dashed')
    ax1.plot(df_eom['SCET_UTC'], df_eom['SSR-B-SBE'], 
             label='SSR_SBEDBE_Data_EOM',
             linestyle='dashed')
    ax1.plot(df_ssr_data['SCET_UTC'], df_ssr_data['SSR-B-SBE'], 
             label='SSR_SBEDBE_Data01312017',
             linestyle='dashed')
    ax1.plot(df_new['datetime'], df_new['SSR-B-SBE'], 
             label='SSR_2004_2016',
             linestyle='dashed',
             linewidth=0.5,
             color='black')
    
    ax1.set_title("SSR-B Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()


def create_merged_file():
    df_2004_2016 = read_ssr_2004_2016()
    df_2004_2016.rename(columns={"datetime": "SCET_UTC"}, inplace=True)
    df_2014_2017 = read_ssr_sbedbe_data('SSR_SBEDBE_Data_EOM.csv')
    df = pd.concat([df_2004_2016, df_2014_2017], axis=0)
    df.drop_duplicates(inplace=True)
    df.to_csv(
        cassini_data_dir + "cassini_ssr_merged.csv",
        sep="\t",
        index=False,
    )  

    return df


def read_cassini_ssr():
    return pd.read_csv(cassini_data_dir + 'cassini_ssr_merged.csv',
    sep='\t', parse_dates=['SCET_UTC'])


def plot_cassini_ssr():
    df = read_cassini_ssr()
    fig, ax1 = plt.subplots(figsize=(10, 6)) 
    
    #ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')
    ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')

    ax1.set_title("SSR-A Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6)) 
    
    #ax1.plot(df['SCET_UTC'], df['SSR-A-SBE'], label='SSR-A-SBE')
    ax1.plot(df['SCET_UTC'], df['SSR-B-SBE'], label='SSR-B-SBE')
    ax1.set_title("SSR-B Single Bit Errors")
    ax1.set_ylabel("Single Bit Error count", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend()
    plt.show()


if __name__ == "__main__":
    create_list_of_apses_times()
    #df = read_cassini_ssr()
    #plot_cassini_ssr()
    #print(df)
    # create_merged_file()
    #df = read_ssr_2004_2016()
    # read_20092010_ssr()
    #filename = '20092010_SSR.csv'
    # read_ssr_with_latradius(filename)
    # plot_together()
    # plot_20092010_ssr()
    #df = read_ssr_with_latradius(filename)
    #print(df)
    # plot_ssr_2004_2016()
    #plot_together()
    # filename = '20042008_SSR_withLatRadius.csv'
    # filename = '20112016_SSR_withlatRadius.csv'
    # df = read_ssr_with_latradius(filename)
    # print(df)
    # plot_ssr_with_latradius(filename)
    
    ### 

    # filename = 'SSR_SBEDBE_Data_EOM.csv'
    # df = read_ssr_sbedbe_data(filename)
    # print(df)
    # plot_ssr_sbedbe(filename)
    
    ### 
    # df = read_ssr_sbedbe_data01312017()
    # plot_ssr_sbedbe_data01312017()

    #     # plot_together()