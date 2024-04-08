import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
#from pandas.plotting import register_matplotlib_converters
#from sqlalchemy import all_
#register_matplotlib_converters()
#from datetime import datetime, timedelta
#import statistics
#from scipy.signal import savgol_filter
#import os
#import sys
#import scipy.stats
#from matplotlib.backends.backend_pdf import PdfPages

## Locked parameterse
#raw_window = 5 # The time bin for calculating the rate for the EDAC curve to be normalized
#smooth_window = 11 # The time bin for calculating the rate for the curve that is to be smoothed
#savgolwindow = 1095  # The size of the window of the Savitzky-Golay filter. 3 years
#polyorder = 3 # Order of the polynomial for Savitzky-Golay filter

## The noise interval
#upper_noiselimit = 0.8885549424705822 # Q3+1.5*IQR 
#lower_noiselimit = -0.7007099616105935 # Q1-0.6*IQR

#upper_boundary_range = 0.18 # Normal distribution
#lower_boundary_range = 0.06 # Normal distribution
#upper_noiselimit =  0.7187784869509977 


def create_rawedac_df(raw_edac_file): # Creates a dataframe from the raw data provided by MEX
    df = pd.read_csv(raw_edac_file,skiprows=15, sep="\t",parse_dates = ['# DATE TIME'])
    columns = list(df.columns.values)
    df.rename(columns={columns[0]: 'date', columns[1]: 'edac'}, inplace=True) 
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by="date")
    df.set_index('date')
    return df 

def read_patched_rawedac(patched_edac):
    df = pd.read_csv(patched_edac,skiprows=0, sep="\t",parse_dates = ['date'])
    return df

def zero_set_correct(raw_edac_file): # Returns the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    ### df = create_rawedac_df(raw_edac_file) ## For not patched EDACs
    df = read_patched_rawedac(raw_edac_file)
    diffs = df.edac.diff() # The difference in counter from row to row
    indices = np.where(diffs<0)[0] #  Finding the indices where the EDAC counter decreases instead of increases or stays the same
    print("This EDAC was zero-set ", len(indices), " number of times.")
    for i in range(0, len(indices)):
        prev_value = df.loc[[indices[i]-1]].values[-1][-1]
        if i == len(indices)-1: # The last time the EDAC counter goes to zero
            df.loc[indices[i]:,'edac'] = df.loc[indices[i]:,'edac'] + prev_value # Add the previous
        else:
            df.loc[indices[i]:indices[i+1]-1,'edac'] = df.loc[indices[i]:indices[i+1]-1,'edac'] + prev_value
    print('Time taken to perform zero-set correction: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    return df 

def resample_corrected_edac(raw_edac_file): # Resamples the zero set corrected EDAC counter to have one reading each day, and save the resulting dataframe to a textfile
    start_time = time.time()
    df = zero_set_correct(raw_edac_file) # Retrieve the zero-set corrected dataframe
    df = df.set_index('date') 
    df_resampled =  df.resample('D').last().ffill()
    df_resampled.reset_index(inplace=True)
    path = 'files/' 
    ### filename = 'resampled_corrected_edac.txt' ### For not-patched EDACs
    filename = 'resampled_corrected_patched_edac.txt'
    df_resampled.to_csv(path + filename, sep='\t', index=False) # Save to file
    print('File ', filename, ' created')
    print('Time taken to resample to daily frequency: ', '{:.2f}'.format(time.time() - start_time) , "seconds")

def read_resampled_df(): # Retrieves the resampled and zerozset corrected edac
    start_time = time.time()
    ### df = pd.read_csv(path +'resampled_corrected_edac.txt',skiprows=0, sep="\t",parse_dates = ['date']) ## For non-patched EDACs
    df = pd.read_csv(path +'resampled_corrected_patched_edac.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    df.set_index('date') # set index to be the dates, instead of 0 1 2 ...
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    print('Time taken to read resampled EDAC to dataframe: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    return df 

def create_rate_df(days_window):
    start_time = time.time()
    df = read_resampled_df() # Fetch resampled EDAC data
    startdate = df['date'][df.index[days_window//2]].date() # The starting date in the data, includes the time
    lastdate = df['date'][df.index[-days_window//2]].date() # The last date and time in the dataset
    print("The starting date is ", startdate, "\nThe last date is ", lastdate)

    df['startwindow_edac'] = df['edac'].shift(days_window//2)
    df['startwindow_date'] = df['date'].shift(days_window//2)
    df['endwindow_date'] = df['date'].shift(-(days_window//2))
    df['endwindow_edac'] = df['edac'].shift(-(days_window//2))

    df['edac_diff'] = df['endwindow_edac'] - df['startwindow_edac']
    df['rate'] = df['edac_diff'] / days_window

 
    new_df = df[['date', 'rate']] # Remove all columns except for the date and the daily rate
    new_df = new_df[new_df['rate'].notna()] # Remove rows without a daily rate

    new_df.to_csv(path + str(days_window)+ '_daily_rate.txt', sep='\t', index=False) # Save to file    
    print("File  ", str(days_window) +"_daily_rate.txt created")
    print('Time taken to create rate_df ', '{:.2f}'.format(time.time() - start_time) , "seconds")

def read_rate_df(days_window):
    start_time = time.time()
    df = pd.read_csv(path + str(days_window)+'_daily_rate.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    df.set_index('date') # set index to be the dates, instead of 0 1 2 ...
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    print('Time taken to read daily rates to dataframe: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    return df 

def remove_spikes_for_smoothing(smooth_window):
    df = read_rate_df(smooth_window) # reads daily rates from a .txt file
    plt.figure()
    plt.plot(df['date'],df['rate'])
    plt.xlabel('Date')
    plt.ylabel('EDAC daily rate')
    plt.grid()
    plt.show()


def show_timerange(startdate, enddate, raw_edac_file):
    startdate_string= str(startdate).replace(" ", "_")
    startdate_string= startdate_string.replace(":", "")
    enddate_string= str(enddate).replace(" ", "_")
    enddate_string = enddate_string.replace(":", "")
    raw_edac =  read_patched_rawedac(raw_edac_file)

    zeroset_edac = zero_set_correct(raw_edac_file)
    rate_df = read_rate_df(5) ## assuming that create_rate_df(days_window) has been run already
    filtered_raw = raw_edac.copy()
    filtered_raw =  filtered_raw[(filtered_raw['date'] > startdate) & (filtered_raw['date'] < enddate)]
    
    filtered_zeroset = zeroset_edac.copy()
    filtered_zeroset = filtered_zeroset[(filtered_zeroset['date'] > startdate) & (filtered_zeroset['date'] < enddate)]
    filtered_rate =  rate_df[(rate_df['date'] > startdate) & (rate_df['date'] < enddate)]
    edac_change = filtered_raw.drop_duplicates(subset='edac', keep='first', inplace=False) # Datetimes where the EDAC is increasing

    filtered_raw.loc[:, 'time_difference'] = filtered_raw['date'].diff().fillna(pd.Timedelta(seconds=0))
    filtered_raw.to_csv(path + 'events/rawEDAC_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False) # Save selected raw EDAC to file
    filtered_rate.to_csv(path + 'events/EDACrate_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)  # Save selected EDAc rate to file
    edac_change.to_csv(path + 'events/EDACchange'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)

    plt.figure()
    plt.plot(rate_df['date'], rate_df['rate'])
    plt.xlabel('Date')
    plt.ylabel('EDAC daily rate')
    plt.grid()
    plt.show()
    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,5))
    
    ax1.scatter(filtered_raw['date'],filtered_raw['edac'], label='Raw EDAC', s=3)
    #ax2.plot(filtered_zeroset['date'], filtered_zeroset['edac'], label ='Zeroset-corrected EDAC')
    ax2.scatter(filtered_rate['date'], filtered_rate['rate'], label ='Daily rate with 5 day window')
    #plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    ax2.set_xlabel('Date', fontsize = 14)
    ax1.set_ylabel('EDAC count', fontsize = 14)
    ax2.set_ylabel('EDAC daily rate', fontsize = 14)
    ax2.tick_params(axis='x', rotation=20)  # Adjust the rotation angle as needed
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.tight_layout(pad=2.0)
    plt.savefig(path+'events/edac_'+startdate_string + '-' + enddate_string + '.png', dpi=300, transparent=False)
    plt.show()


path = 'files/' # Path of where files are located
raw_edac_filename = 'MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt' # Insert the path of the raw EDAC file
patched_edac_filename = 'patched_mex_edac.txt'
raw_window = 5 # The time bin for calculating the rate for the EDAC curve to be normalized
smooth_window = 11 # The time bin for calculating the rate for the curve that is to be smoothed



def main():
    ####### part where you create the txt-files
    ### resample_corrected_edac(path+raw_edac_filename) # For non-patched EDACs
    resample_corrected_edac(path+patched_edac_filename) # This creates the resampled EDAC data file. Only needs to be done once for each file.
    create_rate_df(raw_window) # Creates daily rates based on resampled EDAC. Needs to be only done once.
    #create_rate_df(smooth_window) 

    ####### part where you do stuff
    #remove_spikes_for_smoothing(smooth_window)

    #show_timerange(pd.to_datetime('2017-09-12 23:59:00'), pd.to_datetime('2017-09-14 00:00:00'), path+patched_edac_filename)
    print("wi")
if __name__ == "__main__":
    main()