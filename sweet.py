import numpy as np
import pandas as pd
import time
#import matplotlib.pyplot as plt
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


def create_rawedac_df(raw_edac_filename): # Creates a dataframe from the raw data provided by MEX
    df = pd.read_csv(raw_edac_filename,skiprows=15, sep="\t",parse_dates = ['# DATE TIME'])
    columns = list(df.columns.values)
    df.rename(columns={columns[0]: 'date', columns[1]: 'edac'}, inplace=True) 
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by="date")
    df.set_index('date')
    return df 

def zero_set_correct(raw_edac_filename): # Returns the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    df = create_rawedac_df(raw_edac_filename)
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

def resample_corrected_edac_old(): # Resamples the zero set corrected EDAC counter to have one reading each day, and save the resulting dataframe to a textfile
    start_time = time.time()
    df = zero_set_correct() # Retrieve the zero-set corrected dataframe
    df_list = [] # Initialize list to have the resampled values
    lastdate = df['date'][df.index[-1]].date() # The last date of the EDAC dataset
    currentdate = df['date'][0].date() # Set currentdate to be the first date of the EDAC dataset
    end_reached = False # Variable to be changed to True when the loop has reached the last date available in the EDAC dataset
    print("starting")
    while end_reached != True:
        day_indices = np.where(df['date'].dt.date == currentdate) # The indices in df for where same date as currentdate
        if len(day_indices[0]) == 0: # If date does not exist 
            df_list.append([currentdate, lastreading]) # Let the missing date have the reading of the previous date          
        else:
            lastreading = df['edac'][day_indices[0][-1]]  # The last reading of the current date
            df_list.append([currentdate, lastreading]) # Add the date and the last reading to the list
        currentdate =  currentdate + pd.Timedelta(days=1) # go to next day
        if lastdate-currentdate < pd.Timedelta('0hr0m0s'): # if the currentdate is past the lastdate
            end_reached = True 
    df_resampled =  pd.DataFrame(df_list, columns=['date', 'edac'])
    path = 'files/'
    df_resampled.to_csv(path + "resampled_corrected_edac_old.txt", sep='\t') # Save to file
    print("file created")
    print('Time taken to resample to daily frequency: ', '{:.2f}'.format(time.time() - start_time) , "seconds")




def resample_corrected_edac(raw_edac_filename): # Resamples the zero set corrected EDAC counter to have one reading each day, and save the resulting dataframe to a textfile
    start_time = time.time()
    df = zero_set_correct(raw_edac_filename) # Retrieve the zero-set corrected dataframe
    df = df.set_index('date') 
    df_resampled =  df.resample('D').last().ffill()
    df_resampled.reset_index(inplace=True)
    path = 'files/' 
    filename = 'resampled_corrected_edac.txt'
    df_resampled.to_csv(path + filename, sep='\t') # Save to file
    print('File ', filename, ' created')
    print('Time taken to resample to daily frequency: ', '{:.2f}'.format(time.time() - start_time) , "seconds")



def main():
    raw_edac_filename = 'files/MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt'
    resample_corrected_edac(raw_edac_filename)
if __name__ == "__main__":
    main()