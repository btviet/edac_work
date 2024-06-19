import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

path = 'files/'


def process_rawedac_2022(): # Combines two EDAC files
    rawedac_2020 = 'MEX_EDAC_NDMW0D0G_2020_09_17.txt' # EDAC file 2005-01-01 until 2020-09-17
    rawedac_2020_skiprows = 12
    rawedac_2022 = 'MEX_NDMW0D0G_2022_02_17_16_13_50.116.txt' # EDAC file 2020-09-16 00:00:00.000 2022-02-17 00:00:00.000
    rawedac_2022_skiprows = 15
    df_2020 = pd.read_csv(path+'raw_edac/'+rawedac_2020, skiprows=rawedac_2020_skiprows, sep="\t",parse_dates = ['# DATE TIME'])
    df_2022 = pd.read_csv(path + 'raw_edac/'+rawedac_2022, skiprows=rawedac_2022_skiprows,  sep="\t", parse_dates = ['# DATE TIME']) 
    columns_2020 = list(df_2020.columns.values)
    columns_2022 = list(df_2022.columns.values)
    df_2020.rename(columns={columns_2020[0]: 'datetime', columns_2020[1]: 'edac'}, inplace=True)
    df_2022.rename(columns={columns_2022[0]: 'datetime', columns_2022[1]: 'edac'}, inplace=True)
    df_2020['datetime'] = pd.to_datetime(df_2020['datetime'], format='mixed') #df_2020 has two date formats, changes on 2018-01-18 10:49:54
    df = pd.concat([df_2020, df_2022], ignore_index=True)
    df = df.drop_duplicates() # Overlap on 2020-09-16, and there are duplicate rows in df_2020
    df = df.sort_values(by='datetime')
    df = df.reset_index(drop=True)
    df.to_csv(path + 'raw_edac/mex_edac_2022.txt', sep='\t',date_format='%Y-%m-%d %H:%M:%S.%f', index=False) # Save to file    

def process_rawedac_2024(): # Combines two EDAC data files, one to March 19th, the other to April 11th
    rawedac_2024_03 = 'MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt'
    rawedac_2024_04 = 'MEX_NDMW0D0G_MEX_NACW0D0G_2024_04_11_07_37_22.272.txt' # This file has data from two EDAC counters
    rawedac_2024_03_skiprows = 15
    rawedac_2024_04_skiprows = 16
    df_2024_03 = pd.read_csv(path+'raw_edac/'+rawedac_2024_03,skiprows=rawedac_2024_03_skiprows, sep="\t",parse_dates = ['# DATE TIME'])
    df_2024_04 = pd.read_csv(path+'raw_edac/'+rawedac_2024_04,skiprows=rawedac_2024_04_skiprows, sep="\t",parse_dates=['# DATE TIME'])
    
    columns_2024_03 = list(df_2024_03.columns.values)
    df_2024_03.rename(columns={columns_2024_03[0]: 'datetime', columns_2024_03[1]: 'edac'}, inplace=True)
    columns_2024_04 = list(df_2024_04.columns.values)
    df_2024_04.rename(columns={columns_2024_04[0]: 'datetime', columns_2024_04[1]: 'edac', columns_2024_04[2]: 'edac_nac'}, inplace=True) 
    
    df_2024_04 = df_2024_04[df_2024_04['edac'] != ' '] # Remove the rows where there is no EDAC value
    df_2024_04['edac'] = df_2024_04['edac'].astype(int)
    df_2024_04 = df_2024_04.drop('edac_nac', axis=1) # Remove rows with the other EDAC counter data
    
    #df_2024_03.to_csv(path + 'raw_edac/df_2024_03.txt', sep='\t',date_format='%Y-%m-%d %H:%M:%S.%f', index=False) # Save to .txt file    
    #df_2024_04.to_csv(path + 'raw_edac/df_2024_04.txt', sep='\t',date_format='%Y-%m-%d %H:%M:%S.%f', index=False) # Save to .txtfile

    ########### Look for missing values ########
    '''
    startdate = pd.to_datetime('2024-02-20 00:05:12.959000') # The time period where the EDAC data overlaps in the two data sets
    enddate = pd.to_datetime('2024-03-18 12:25:29.487000')
    df_2024_03 =  df_2024_03[(df_2024_03['datetime'] > startdate) & (df_2024_03['datetime'] < enddate)]
    df_2024_04 =  df_2024_04[(df_2024_04['datetime'] > startdate) & (df_2024_04['datetime'] < enddate)]
    merged = pd.merge(df_2024_03, df_2024_04, how='outer', indicator=True)
    both_in_both = merged[merged['_merge'] == 'both'] # Select rows that exist in both df1 and df2
    print(both_in_both)
    only_in_df_2024_03 = merged[merged['_merge'] == 'left_only']  # Select rows that are only in df_2024_03
    print(only_in_df_2024_03)
    only_in_df_2024_04 = merged[merged['_merge'] == 'right_only']     # Select rows that are only in df_2024_04 
    print(only_in_df_2024_04)'''

    combined_df = pd.concat([df_2024_03, df_2024_04])
    combined_df.drop_duplicates(subset='datetime', keep='first', inplace=True)
    combined_df = combined_df.sort_values(by='datetime')
    combined_df = combined_df.reset_index(drop=True)
    combined_df.to_csv(path + 'raw_edac/mex_edac_2024.txt', sep='\t',date_format='%Y-%m-%d %H:%M:%S.%f', index=False) # Save to file    


def combine_rawedac_2022_2024():
    df_2022 = pd.read_csv(path + 'raw_edac/mex_edac_2022.txt', sep="\t",parse_dates = ['datetime'])
    df_2024 =pd.read_csv(path + 'raw_edac/mex_edac_2024.txt', sep="\t",parse_dates = ['datetime'])
  
    ###### Patch in January 18th, 2018 - February 17th, 2022 from the old EDAC data
    relevant_old_raw = df_2022[(df_2022['datetime'] > pd.to_datetime('2018-01-18 10:41:22')) & (df_2022['datetime'] < pd.to_datetime('2022-02-16 23:51:59'))] #
 
    combined_df = pd.concat([relevant_old_raw, df_2024])

    combined_df.drop_duplicates(subset='datetime', keep='first', inplace=True)
    df_sorted = combined_df.sort_values(by='datetime')
    df_sorted.to_csv(path + 'raw_edac/patched_mex_edac.txt', sep='\t', index=False) # Save to file   
     
    ##### Check where the new EDAC (df_2024) is missing values
    '''
    startdate = pd.to_datetime('2022-01-26')
    enddate= pd.to_datetime('2022-02-15')
    filtered_df_2024 = df_2024[(df_2024['datetime'] > startdate) & (df_2024['datetime'] < enddate)]
    filtered_df_2022= df_2022[(df_2022['datetime'] > startdate) & (df_2022['datetime'] <enddate)]
    filtered_combined_df =  combined_df[(combined_df['datetime'] > startdate) & (combined_df['datetime'] < enddate)]
    plt.figure()
    plt.scatter(filtered_df_2024['datetime'], filtered_df_2024['edac'])
    plt.title('Newest EDAC')
    plt.show()

    plt.figure()
    plt.scatter(filtered_df_2022['datetime'], filtered_df_2022['edac'])
    plt.title('Old EDAC')
    plt.show()

    plt.figure()
    plt.scatter(filtered_df_2022['datetime'], filtered_df_2022['edac'], label = 'old', c ='skyblue', marker="o", s=10)
    plt.scatter(filtered_df_2024['datetime'], filtered_df_2024['edac'], label ='new',c='deeppink', marker="o", s=30)
    plt.ylabel('EDAC count')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(filtered_combined_df['datetime'], filtered_combined_df['edac'])
    plt.title('Combined EDAC')
    plt.show()
    '''




def main():
    process_rawedac_2022()
    process_rawedac_2024()
    combine_rawedac_2022_2024()

if __name__ == "__main__":
    main()