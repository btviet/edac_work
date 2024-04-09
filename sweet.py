import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
import scipy.optimize
from lmfit.models import SkewedGaussianModel
#from matplotlib.gridspec import GridSpec
#from pandas.plotting import register_matplotlib_converters
#from sqlalchemy import all_
#register_matplotlib_converters()

#import statistics

#import os
#import sys
#import scipy.stats
#from matplotlib.backends.backend_pdf import PdfPages

## Locked parameters
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


def read_rawedac_df(raw_edac_file): # Creates a dataframe from the raw data provided by MEX
    df = pd.read_csv(raw_edac_file,skiprows=15, sep="\t",parse_dates = ['# DATE TIME'])
    columns = list(df.columns.values)
    df.rename(columns={columns[0]: 'datetime', columns[1]: 'edac'}, inplace=True) 
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by="datetime")
    df.set_index('datetime')
    return df 

def read_patched_rawedac(patched_edac_path): # Reads the patched MEX EDAC
    df = pd.read_csv(patched_edac_path,skiprows=0, sep="\t",parse_dates = ['datetime'])
    return df

def create_zero_set_correct(raw_edac_file): # Returns the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    print("--------- Starting the zeroset correction ---------")
    ### df = read_rawedac_df(raw_edac_file) ## For not patched EDACs
    df = read_patched_rawedac(raw_edac_file)
    diffs = df.edac.diff() # The difference in counter from row to row
    indices = np.where(diffs<0)[0] #  Finding the indices where the EDAC counter decreases instead of increases or stays the same
    print("The EDAC data was zero-set ", len(indices), " times.")
    for i in range(0, len(indices)):
        prev_value = df.loc[[indices[i]-1]].values[-1][-1]
        if i == len(indices)-1: # The last time the EDAC counter goes to zero
            df.loc[indices[i]:,'edac'] = df.loc[indices[i]:,'edac'] + prev_value # Add the previous
        else:
            df.loc[indices[i]:indices[i+1]-1,'edac'] = df.loc[indices[i]:indices[i+1]-1,'edac'] + prev_value
    df.to_csv(path + 'zerosetcorrected_edac.txt', sep='\t', index=False) # Save to file    
    print("File  ", "zerosetcorrected_edac.txt created")
    print('Time taken to perform zero-set correction and create files: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    return df 

def read_zero_set_correct():
    df = pd.read_csv(
        path+'zerosetcorrected_edac.txt',skiprows=0, sep="\t",parse_dates = ['datetime'])
    return df

def create_resampled_corrected_edac(zerosetcorrected_df): # Resamples the zero set corrected EDAC counter to have one reading each day, and save the resulting dataframe to a textfile
    start_time = time.time()
    print('--------- Starting the resampling to a daily frequency process ------')
    zerosetcorrected_df = zerosetcorrected_df.set_index('datetime') 
    df_resampled =  zerosetcorrected_df.resample('D').last().ffill()
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'datetime': 'date'}, inplace=True)
    ### filename = 'resampled_corrected_edac.txt' ### For not-patched EDACs
    filename = 'resampled_corrected_patched_edac.txt'
    df_resampled.to_csv(path + filename, sep='\t', index=False) # Save to file
    print('File ', filename, ' created')
    print('Time taken to resample to daily frequency and create files: ', '{:.2f}'.format(time.time() - start_time) , "seconds")

def read_resampled_df(): # Retrieves the resampled and zerozset corrected edac
    #start_time = time.time()
    ### df = pd.read_csv(path +'resampled_corrected_edac.txt',skiprows=0, sep="\t",parse_dates = ['date']) ## For non-patched EDACs
    df = pd.read_csv(path +'resampled_corrected_patched_edac.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    df.set_index('date') # set index to be the dates, instead of 0 1 2 ...
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    #print('Time taken to read resampled EDAC to dataframe: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    return df 

def create_rate_df(days_window):
    start_time = time.time()
    print("--------- Calculating the daily rates ---------" )
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
    #start_time = time.time()
    df = pd.read_csv(path + str(days_window)+'_daily_rate.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    df.set_index('date') # set index to be the dates, instead of 0 1 2 ...
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    #print('Time taken to read daily rates to dataframe: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    df =df[(df['date'] < pd.to_datetime('2022-02-14 23:59:00'))] # valid timeframe
    return df 

def fit_sin(tt, yy): # Helping function for gcr_edac()
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
    
def gcr_edac():
    rate_df = read_rate_df(smooth_window)
    start_date = rate_df['date'].iloc[0]
    tt = np.array([(x - start_date).days for x in rate_df['date']])
    yy = rate_df['rate']
    res = fit_sin(tt, yy)
    x_datetime = np.array([start_date + pd.Timedelta(days=x) for x in tt])
    df = pd.DataFrame({'date':x_datetime,'rate':  res["fitfunc"](tt)})
    return df

def create_normalized_rates(): # Return the normalized EDAC rate
    gcr_component = gcr_edac()
    first_gcr = gcr_component['date'].iloc[0]
    last_gcr = gcr_component['date'].iloc[-1]
    rate_df = read_rate_df(raw_window)
    first_rate = rate_df['date'].iloc[0]
    last_rate = rate_df['date'].iloc[-1]

    if first_gcr >= first_rate:
        rate_df =  rate_df[rate_df['date'] >= first_gcr]
    else:
        gcr_component = gcr_component[gcr_component['date'] >= first_rate]

    if last_gcr >= last_rate:
        gcr_component = gcr_component[gcr_component['date'] <= last_rate]
        
    else:
        rate_df = rate_df[rate_df['date'] <= last_gcr]

    rate_df.reset_index(drop=True, inplace=True)
    gcr_component.reset_index(drop=True, inplace=True)

    normalized_df = rate_df.copy()
    normalized_df['gcr_component'] = gcr_component['rate']
    normalized_df['normalized_rate'] = normalized_df['rate']/normalized_df['gcr_component']
    
    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,6))
    
    ax1.plot(normalized_df['date'],normalized_df['rate'], label='EDAC daily rate, ' + str(raw_window) + ' day swindow')
    ax1.plot(normalized_df['date'],normalized_df['gcr_component'], label='GCR component of EDAC daily rate')
    ax2.plot(normalized_df['date'], normalized_df['normalized_rate'], label='Normalized daily EDAC rate')
    ax1.set_ylim([-0.5, 7])
    ax2.set_ylim([-0.5, 7])
    #plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    ax2.set_xlabel('Date', fontsize = 12)
    ax1.set_ylabel('EDAC daily rate', fontsize = 12)
    ax2.set_ylabel('EDAC normalized daily rate', fontsize = 12)
    ax2.tick_params(axis='x', rotation=20)  # Adjust the rotation angle as needed
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.tight_layout(pad=1.0)
    #plt.savefig(path+'events/edac_'+startdate_string + '-' + enddate_string + '.png', dpi=300, transparent=False)
    plt.show()
    normalized_df.to_csv(path + 'normalized_edac.txt', sep='\t', index=False) # Save to file

def read_normalized_rates():
    df = pd.read_csv(path + 'normalized_edac.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    return df
### -------------------------------------------
def print_missing_dates(date_column):
    ## example of datetime_column: df['datetime].dt.date  # remove the time from datetime object
    start_date = date_column.iloc[0]
    end_date = date_column.iloc[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    missing_dates = date_range[~date_range.isin(date_column)]
    print("Start date is ", start_date, ". End date is ", end_date)
    print("Missing dates: ", missing_dates)

def show_timerange(startdate, enddate, raw_edac_file):
    startdate_string= str(startdate).replace(" ", "_")
    startdate_string= startdate_string.replace(":", "")
    enddate_string= str(enddate).replace(" ", "_")
    enddate_string = enddate_string.replace(":", "")
    raw_edac =  read_patched_rawedac(raw_edac_file)

    zeroset_edac = read_zero_set_correct()
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

def plot_rates_all(rate_df):
    plt.figure()
    plt.plot(rate_df['date'], rate_df['rate'])
    plt.xlabel('Date')
    plt.ylabel('EDAC daily rate')
    plt.grid()
    plt.show()

def plot_rate_distribution():
    df = read_normalized_rates()
    binsize = 0.2
    max_rate = np.max(df['normalized_rate'])
    min_rate = np.min(df['normalized_rate'])
    bins = np.arange(min_rate, max_rate+binsize, binsize) # Choose the size of the bins
    counts, bin_edges = np.histogram(df['normalized_rate'], bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    model = SkewedGaussianModel()   # Create a SkewedGaussianModel
    params = model.guess(counts, x=bin_centers) # Guess initial parameters
    result = model.fit(counts, params, x=bin_centers) # Fit the model to the data

    print(result.fit_report())
    fitted_params = result.params.valuesdict()
    
    # Plot the data and the fitted model
    plt.figure()
    #plt.plot(bin_centers, counts, label='Data')
    plt.hist(df['normalized_rate'],bins=bins, density=False, ec='black')
    plt.plot(bin_centers, result.best_fit, label='Fitted model')
    plt.legend()
    plt.xlabel('Normalized EDAC daily rate')
    plt.ylabel('Occurrences')
    plt.title('Fitting with Skewed Gaussian Model')
    plt.show()

path = 'files/' # Path of where files are located
raw_edac_filename = 'MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt' # Insert the path of the raw EDAC file
patched_edac_filename = 'patched_mex_edac.txt'
raw_window = 5 # The time bin for calculating the rate for the EDAC curve to be normalized
smooth_window = 11 # The time bin for calculating the rate for the curve that is to be smoothed



def process_new_raw_edac(): # Creates .txt files based on the raw EDAC. Do only once
    create_zero_set_correct(path+patched_edac_filename) # Create zeroset corrected EDAC file, needs to be done only once.
    zerosetcorrected_df = read_zero_set_correct() # Read the zeroset corrected file
    create_resampled_corrected_edac(zerosetcorrected_df) # Resample to a daily frequency. Needs to be done only once.
    create_rate_df(raw_window) # Creates daily rates based on resampled EDAC.
    create_rate_df(smooth_window) # Creates daily rates based on resampled EDAC
    
def main():

    #process_new_raw_edac()
    #print_missing_dates(zerosetcorrected_df['datetime'].dt.date)
    ####### part where you do stuff
    #remove_spikes_for_smoothing(smooth_window)

    #show_timerange(pd.to_datetime('2017-09-12 23:59:00'), pd.to_datetime('2017-09-14 00:00:00'), path+patched_edac_filename)
    #create_normalized_rates()
    #read_normalized_rates()

    plot_rate_distribution()
    print("End")
if __name__ == "__main__":
    main()