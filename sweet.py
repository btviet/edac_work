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

def zero_set_correct(raw_edac_file): # Returns the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    df = create_rawedac_df(raw_edac_file)
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
    filename = 'resampled_corrected_edac.txt'
    df_resampled.to_csv(path + filename, sep='\t') # Save to file
    print('File ', filename, ' created')
    print('Time taken to resample to daily frequency: ', '{:.2f}'.format(time.time() - start_time) , "seconds")


def creating_df(): # Retrieves the zero set corrected raw EDAC counter
    start_time = time.time()
    df = pd.read_csv(path +'resampled_corrected_edac.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    df.set_index('date') # set index to be the dates, instead of 0 1 2 ...
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    print(df)
    print('Time taken to read resampled EDAC to dataframe: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    return df 


def create_rate_df_old(days_window): # Function to calculate a rate for each date
    start_time = time.time()
    # day window is the an odd number
    df_list = [] # Initialize list to keep dates and daily rates
    df = creating_df() # df with date and the value of the EDAC counter
    startdate = df['date'][df.index[days_window//2]].date() # The starting date in the data, includes the time
    lastdate = df['date'][df.index[-1]].date() # The last date and time in the dataset
    print("The starting date is ", startdate, "\nThe last date is ", lastdate)
    currentdate = startdate # Start from the first date
    end_reached = False
    while not end_reached:
        beginning_window = (currentdate-pd.Timedelta(days=days_window//2)) # Place the current date to be the middle of the window
        end_window = (currentdate + pd.Timedelta(days=days_window//2))
        beginning_values = df.iloc[np.where(df['date'].dt.date == beginning_window)[0][0]] # Date and EDAC value of beginning window
        end_values = df.iloc[np.where(df['date'].dt.date == end_window)[0][0]] # Date and EDAC of ending window
        diff = end_values-beginning_values
        diff_edac = diff[1] # Difference in EDAC value
        diff_days = diff[0] # Difference in days
        number_of_days = diff_days.days+1 # Number of days in the window
        current_edac_rate = diff_edac/number_of_days # Calculate the EDAC rate
        df_list.append([currentdate, current_edac_rate])
        currentdate = currentdate + pd.Timedelta(days=1) # Iterate to next day
        if lastdate-currentdate < pd.Timedelta(days=days_window//2): # Stopping condition for while-loop
            end_reached = True
        if currentdate.year != (currentdate-pd.Timedelta(days=1)).year: # For observation while running code
            print("Year reached: ", currentdate.year)
    df_rate =  pd.DataFrame(df_list, columns=['date', 'rate']) # Convert the list to a dataframe
    print('Time taken to create rate_df ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    ## Initial time was around 33 seconds. 
    return df_rate # Return date, rate dataframe



def create_rate_df(days_window):
    start_time = time.time()
    df = creating_df() # Fetch resampled EDAC data
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

    new_df.to_csv(path + str(days_window)+ '_daily_rate.txt', sep='\t') # Save to file    
    print("File  ", str(days_window) +"_daily_rate.txt created")
    print('Time taken to create rate_df ', '{:.2f}'.format(time.time() - start_time) , "seconds")


path = 'files/' # Path of where files are located
raw_edac_filename = 'MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt' # Insert the path of the raw EDAC file
raw_window = 5 # The time bin for calculating the rate for the EDAC curve to be normalized

def main():

    #resample_corrected_edac(path+raw_edac_filename) # This creates the resampled EDAC data file. Only needs to be done once for each file.
    create_rate_df(raw_window)
if __name__ == "__main__":
    main()