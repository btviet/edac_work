import numpy as np
import pandas as pd
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


''' Zero set correction'''
def create_rawedac_df(): # Creates a dataframe from the raw data provided by MEX
    path = 'files/' # The location where the EDAC files are
    df = pd.read_csv('files/MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt',skiprows=12, sep="\t",parse_dates = ['# DATE TIME'])
    df.rename(columns={'# DATE TIME': 'date', 'NDMW0D0G [MEX]': 'edac'}, inplace=True) # Changing the name of the columns, old_name: new_name
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by="date")
    df.set_index('date')
    return df 


def zero_set_correct(): # Returns the zero-set corrected dataframe of the raw EDAC counter
    df = create_rawedac_df()
    diffs = df.edac.diff() # The difference in counter from row to row
    indices = np.where(diffs<0)[0] #  Finding the indices where the EDAC counter decreases instead of increases or stays the same
    for i in range(0, len(indices)):
        prev_value = df.loc[[indices[i]-1]].values[-1][-1]
        if i == len(indices)-1: # The last time the EDAC counter goes to zero
            df.loc[indices[i]:,'edac'] = df.loc[indices[i]:,'edac'] + prev_value # Add the previous
        else:
            df.loc[indices[i]:indices[i+1]-1,'edac'] = df.loc[indices[i]:indices[i+1]-1,'edac'] + prev_value
    return df 



print(zero_set_correct())