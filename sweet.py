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
    df = pd.read_csv(path + '/raw_files/MEX_EDAC.txt',skiprows=12, sep="\t",parse_dates = ['# DATE TIME'])
    df2 = pd.read_csv(path + '/raw_files/MEX_NDMW0D0G_2022_02_17_16_13_50.116.txt', skiprows=15,  sep="\t",parse_dates = ['# DATE TIME'])
    df.rename(columns={'# DATE TIME': 'date', 'NDMW0D0G [MEX]': 'edac'}, inplace=True) # Changing the name of the columns, old_name: new_name
    df2.rename(columns={'# DATE TIME': 'date', 'NDMW0D0G - AVG - 1 Non [MEX]': 'edac'}, inplace=True) # Changing the name of the columns, old_name: new_name
    df = df.append(df2)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by="date")
    df.set_index('date')
    return df 