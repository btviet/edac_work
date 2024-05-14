import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
import scipy.optimize
from lmfit.models import SkewedGaussianModel
from scipy import stats


def read_rawedac(raw_edac_path): # Reads the patched MEX EDAC
    df = pd.read_csv(raw_edac_path,skiprows=0, sep="\t",parse_dates = ['datetime'])
    return df

def create_zero_set_correct(raw_edac_path): # Returns the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    print("--------- Starting the zeroset correction ---------")
    df = read_rawedac(raw_edac_path)
    diffs = df.edac.diff() # The difference in counter from row to row
    indices = np.where(diffs<0)[0] #  Finding the indices where the EDAC counter decreases instead of increases or stays the same
    print("This EDAC data set was zero-set ", len(indices), " times.")
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

def create_resampled_edac(zerosetcorrected_df,method): # Resamples the zero set corrected EDAC counter to have one reading each day, and save the resulting dataframe to a textfile
    start_time = time.time()
    print('--------- Starting the resampling to a daily frequency process ------')
    zerosetcorrected_df = zerosetcorrected_df.set_index('datetime') 
    if method == 'roll':
        # Does not include the daily rate in the .txt file
        df_resampled =  zerosetcorrected_df.resample('D').last().ffill()
        df_resampled.reset_index(inplace=True)
        df_resampled.rename(columns={'datetime': 'date'}, inplace=True)
        df_resampled['date'] = df_resampled['date'] + pd.Timedelta(hours=12) # Set the value to be at noon
        filename = 'resampled_corrected_edac_roll.txt'
        df_resampled.to_csv(path + filename, sep='\t', index=False) # Save to file
        print('File ', filename, ' created')
        print('Time taken to resample to daily frequency and create files: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    else:
        last_df = zerosetcorrected_df.resample('D').last().ffill() # keeps the last row for each day
        df_resampled = zerosetcorrected_df.resample('D').first().ffill() # keeps the first row for each day
        df_resampled.reset_index(inplace=True) 
        last_df.reset_index(inplace=True)
        df_resampled['edac_last'] = last_df['edac']
        df_resampled.rename(columns={'datetime': 'date', 'edac':'edac_first'}, inplace=True)
        df_resampled['daily_rate'] = df_resampled['edac_last']-df_resampled['edac_first']
        df_resampled['date'] = df_resampled['date']+pd.Timedelta(hours=12) # set datetime of each date to noon
        filename = 'resampled_corrected_edac.txt'
        df_resampled.to_csv(path + filename, sep='\t', index=False) # Save to file
        print('File ', filename, ' created')
        print('Time taken to resample to daily frequency, calculate daily rate and create files: ', '{:.2f}'.format(time.time() - start_time) , "seconds")

def read_resampled_df(method): # Retrieves the resampled and zeroset corrected edac

    path='files/'
    if method =='roll':
        filename = 'resampled_corrected_edac_roll.txt'
    else:
        filename ='resampled_corrected_edac.txt'
    df = pd.read_csv(path +filename,skiprows=0, sep="\t",parse_dates = ['date'])
    return df 

def create_rate_df(days_window): # rolling window method
    start_time = time.time()
    print("--------- Calculating the daily rates ---------" )
    df = read_resampled_df('roll') # Fetch resampled EDAC data, read output from create_resampled_corrected_edac()
    startdate = df['date'][df.index[days_window//2]].date() # The starting date in the data
    lastdate = df['date'][df.index[-days_window//2]].date() # The last date and time in the dataset
    print("The starting date is ", startdate, "\nThe last date is ", lastdate)

    df['startwindow_edac'] = df['edac'].shift(days_window//2)
    df['startwindow_date'] = df['date'].shift(days_window//2)
    df['endwindow_date'] = df['date'].shift(-(days_window//2))
    df['endwindow_edac'] = df['edac'].shift(-(days_window//2))

    df['edac_diff'] = df['endwindow_edac'] - df['startwindow_edac']
    df['daily_rate'] = df['edac_diff'] / days_window

    new_df = df[['date', 'daily_rate']] # Remove all columns except for the date and the daily rate
    new_df = new_df[new_df['daily_rate'].notna()] # Remove rows without a daily rate

    new_df.to_csv(path + str(days_window)+ '_daily_rate.txt', sep='\t', index=False) # Save to file    
    print("File  ", str(days_window) +"_daily_rate.txt created")
    print('Time taken to create rate_df ', '{:.2f}'.format(time.time() - start_time) , "seconds")

def read_rate_df(days_window): # rolling window method
    #start_time = time.time()
    df = pd.read_csv(path + str(days_window)+'_daily_rate.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    df.set_index('date') # set index to be the dates, instead of 0 1 2 ...
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    #print('Time taken to read daily rates to dataframe: ', '{:.2f}'.format(time.time() - start_time) , "seconds")
    #df =df[(df['date'] < pd.to_datetime('2022-02-14 23:59:00'))] # valid timeframe
    return df 

def fit_sine_gcr(method): # nonroll daily rate
    if method == 'roll':
        rate_df = read_rate_df(smooth_window)
    else:
        rate_df = read_resampled_df(method)
    start_date = rate_df['date'].iloc[0]
    last_date = rate_df['date'].iloc[-1]
    ###rate_df = rate_df[rate_df['daily_rate'] > 0] # Remove the rows with rate equal to 0
    tt = np.array([(x - start_date).days for x in rate_df['date']])
    yy = rate_df['daily_rate']
    res = fit_sin(tt, yy)
    fitfunc = res["fitfunc"]
    x_datetime = np.array([start_date + pd.Timedelta(days=x) for x in tt])
    date_range = pd.date_range(start=start_date, end=last_date)
    #fit_values = fitfunc(np.arange(len(date_range)))
    #df = pd.DataFrame({'date': date_range, 'daily_rate': fit_values})
 
    rate_df['fit'] = res["fitfunc"](tt)

    '''
    df_all = pd.DataFrame({'date':x_datetime,'rate_gcr':  res["fitfunc"](tt)}) #sine fit with all data
    plt.figure()
    plt.plot(rate_df['date'],rate_df['daily_rate'], label='EDAC count rate')
    plt.plot(df_all['date'], df_all['rate_gcr'], label='sine fit with all data')
    plt.plot(rate_df['date'], rate_df['sine_fit'], label='Sine fit when excluding 0s')
    plt.legend()
    plt.ylim([0, 18])
    plt.grid()
    plt.show()
    '''
    #return df_low
    return rate_df

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
    #print(A, w, p, c, f)
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def create_detrended_rates(method):
    gcr_component = savitzky_fit_gcr(method)
    first_gcr = gcr_component['date'].iloc[0]
    last_gcr = gcr_component['date'].iloc[-1]

    if method == 'roll':
        rate_df = read_rate_df(raw_window)
    else:
        rate_df = read_resampled_df(method)

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

    detrended_df = rate_df.copy()

    detrended_df['gcr_component'] = gcr_component['fit']
    
    ### Detrending by subtraction
    detrended_df['detrended_rate'] = detrended_df['daily_rate']-detrended_df['gcr_component']
    print("detrended_df: ", detrended_df)

    ### Detrending by division
    #detrended_df['detrended_rate'] = detrended_df['daily_rate']/detrended_df['gcr_component']
    
    start_date = datetime.strptime('2004-01-01',"%Y-%m-%d")
    suncolor = 'red' #'#4daf4a'
    df_sun = process_sidc_ssn()
    index_exact =  np.where((df_sun['date']==start_date))[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed= savgol_filter(df_sun['daily_sunspotnumber'], savgolwindow_sunspots, 3)

    fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, figsize=(10,8))
    
    ax1.plot(detrended_df['date'],detrended_df['daily_rate'], label='EDAC daily rate')
    ax1.plot(detrended_df['date'],detrended_df['gcr_component'], label='GCR component of EDAC daily rate')
    ax2.plot(detrended_df['date'], detrended_df['detrended_rate'], color='#4daf4a',label='Detrended EDAC rate')
    ax3.plot(df_sun['date'],df_sun['daily_sunspotnumber'], color='#f781bf',label="Number of sunspots")
    ax3.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,
             color='#a65628',
             label='Smoothed sunspots')
    ax3.set_ylabel('Sunspot number')
    ax1.set_ylim([-0.5, 18])
    #ax2.set_ylim([-0.5, 18])
    #plt.suptitle('Normalization by subtraction', fontsize=16)
    ax3.set_xlabel('Date', fontsize = 12)
    ax1.set_ylabel('EDAC daily rate', fontsize = 12)
    ax2.set_ylabel('Detrended rate', fontsize = 12)
    ax2.tick_params(axis='x', rotation=20)  # Adjust the rotation angle as needed
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout(pad=1.0)
    #plt.savefig(path+'events/edac_'+startdate_string + '-' + enddate_string + '.png', dpi=300, transparent=False)
    plt.show()
    if method == 'roll':
        filename = 'detrended_edac_roll.txt'
    else:
        filename = 'detrended_edac_nonroll.txt'
    detrended_df.to_csv(path + filename, sep='\t', index=False) # Save to file
    print("File ", filename, " created")

def create_standardized_rates(method):
    detrended_df = read_detrended_rates(method)
    detrended_mean = detrended_df['detrended_rate'].mean()
    detrended_std = detrended_df['detrended_rate'].std()
    detrended_df['standardized_rate'] = (detrended_df['detrended_rate']-detrended_mean)/detrended_std
    print(detrended_df)
    print("Mean, std: ", detrended_mean, detrended_std)

    start_date = datetime.strptime('2004-01-01',"%Y-%m-%d")
    suncolor = 'red' #'#4daf4a'
    df_sun = process_sidc_ssn()
    index_exact =  np.where((df_sun['date']==start_date))[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed= savgol_filter(df_sun['daily_sunspotnumber'], savgolwindow_sunspots, 3)

    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,8))
    
    ax1.plot(detrended_df['date'], detrended_df['standardized_rate'], color='#4daf4a',label='Standardized de-trended EDAC rate')
    ax2.plot(df_sun['date'],df_sun['daily_sunspotnumber'], color='#f781bf',label="Number of sunspots")
    ax2.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,
             color='#a65628',
             label='Smoothed sunspots')
    ax2.set_ylabel('Sunspot number')

    ax2.set_xlabel('Date', fontsize = 12)
    ax1.set_ylabel('Standardized rate', fontsize = 12)
    ax2.tick_params(axis='x', rotation=20)  # Adjust the rotation angle as needed
    ax1.grid()
    ax2.grid()

    ax1.legend()
    ax2.legend()

    plt.tight_layout(pad=1.0)
    #plt.savefig(path+'events/edac_'+startdate_string + '-' + enddate_string + '.png', dpi=300, transparent=False)
    plt.show()
    if method == 'roll':
        filename = 'standardized_edac_roll.txt'
    else:
        filename = 'standardized_edac_nonroll.txt'
    detrended_df.to_csv(path + filename, sep='\t', index=False) # Save to file
    print("File ", filename, " created")



def create_normalized_rates(method): # Calculate normalized EDAC rate and save to .txt file, roll
    start_time = time.time()
    print('--------- Starting the normalization process ------')
    #gcr_component = fit_sine_gcr(method)
    gcr_component = savitzky_fit_gcr(method)
    first_gcr = gcr_component['date'].iloc[0]
    last_gcr = gcr_component['date'].iloc[-1]

    if method == 'roll':
        rate_df = read_rate_df(raw_window)
    else:
        rate_df = read_resampled_df(method)

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

    normalized_df['gcr_component'] = gcr_component['fit']
    ## DIVISION
    #normalized_df['normalized_rate'] = normalized_df['daily_rate']/normalized_df['gcr_component']
    ## SUBTRACTION
    std_dev = normalized_df['daily_rate'].std()
    #print("std_dev: ", std_dev)
    #normalized_df['normalized_rate'] = (normalized_df['daily_rate']-normalized_df['gcr_component'])/std_dev
    normalized_df['normalized_rate'] = normalized_df['daily_rate']-normalized_df['gcr_component']
    print("normalized_df: ", normalized_df)
    print("detrended mean: ", normalized_df['normalized_rate'].mean())
    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,6))
    
    ax1.plot(normalized_df['date'],normalized_df['daily_rate'], label='EDAC daily rate')
    ax1.plot(normalized_df['date'],normalized_df['gcr_component'], label='GCR component of EDAC daily rate')
    ax2.plot(normalized_df['date'], normalized_df['normalized_rate'], color='#4daf4a',label='Normalized daily EDAC rate')
    
    ax1.set_ylim([-0.5, 18])
    #ax2.set_ylim([-0.5, 18])
    plt.suptitle('Normalization by subtraction', fontsize=16)
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
    if method == 'roll':
        filename = 'normalized_edac_roll.txt'
    else:
        filename = 'normalized_edac_nonroll.txt'
    normalized_df.to_csv(path + filename, sep='\t', index=False) # Save to file
    print("File ", filename, " created")

def savitzky_fit_gcr(method):
    rate_df = read_resampled_df(method)
    #rate_df = rate_df[rate_df['daily_rate'] > 0] # Remove zero days
    savgolwindow=500
    polyorder=2
    y_filtered = savgol_filter(rate_df['daily_rate'], savgolwindow, polyorder) # Apply filtering to the EDAC rates with large spikes removed

    rate_df['fit'] = y_filtered
    start_date = datetime.strptime('2004-01-01',"%Y-%m-%d")
    '''
    df_sun = process_sidc_ssn()
    index_exact =  np.where((df_sun['date']==start_date))[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed= savgol_filter(df_sun['daily_sunspotnumber'], savgolwindow_sunspots, 3)
    #####rate_df['normalized']=rate_df['daily_rate']/y_filtered
    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,6))

    ax1.plot(rate_df['date'],rate_df['daily_rate'],label='EDAC count rate')
    ax1.plot(rate_df['date'],y_filtered,label='Savitzky-Golay fit')
    ax2.set_xlabel('Date')
    ax1.set_ylabel('EDAC count rate')
    ax2.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,
             color='#4daf4a',
             label='Smoothed sunspots')
    ax2.set_ylabel('Sunspot number')
    #ax2.set_ylabel('EDAC normalized rate')
    #ax2.plot(rate_df['date'],rate_df['normalized'],label='Normalized')
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.show()

    #binsize = 0.5
    #max = np.max(rate_df['normalized'])
    #min = np.min(rate_df['normalized'])
    #bins = np.arange(min, max+binsize, binsize) # Choose the size of the bins
    #plt.figure(figsize=(8,6))
    #result =plt.hist(rate_df['normalized'],bins = bins, density = False, color='#FF6B6B',ec='black')
    #plt.xlabel('EDAC normalized rate')
    #plt.ylabel('Occurrences')
    #plt.show()
    '''
    return rate_df

def read_normalized_rates(method):
    if method == 'roll':
        df = pd.read_csv(path + 'normalized_edac_roll.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    else:
        df = pd.read_csv(path + 'normalized_edac_nonroll.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    return df

def read_detrended_rates(method):
    if method == 'roll':
        df = pd.read_csv(path + 'detrended_edac_roll.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    else:
        df = pd.read_csv(path + 'detrended_edac_nonroll.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    return df

def read_standardized_rates(method):
    if method == 'roll':
        df = pd.read_csv(path + 'standardized_edac_roll.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    else:
        df = pd.read_csv(path + 'standardized_edac_nonroll.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    return df 
### -------------------------------------------
def print_missing_dates(date_column):
    ## example of date_column input: df['datetime].dt.date  # remove the time from datetime object
    start_date = date_column.iloc[0]
    end_date = date_column.iloc[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    missing_dates = date_range[~date_range.isin(date_column)]
    print("Start date is ", start_date, ". End date is ", end_date)
    print("Missing dates: ", missing_dates)
    print(len(missing_dates))

def show_timerange(startdate, enddate, raw_edac_file, method):
    raw_edac = read_rawedac(path+raw_edac_file)
    filtered_raw = raw_edac.copy()
    filtered_raw =  filtered_raw[(filtered_raw['datetime'] > startdate) & (filtered_raw['datetime'] < enddate)]
    
    zeroset_edac = read_zero_set_correct()
    filtered_zeroset = zeroset_edac.copy()
    filtered_zeroset = filtered_zeroset[(filtered_zeroset['datetime'] > startdate) & (filtered_zeroset['datetime'] < enddate)]

    normalized_rate = read_normalized_rates(method)
    
    normalized_rate =  normalized_rate[(normalized_rate['date'] > startdate) & (normalized_rate['date'] < enddate)]
    print("normalized_rate: ", normalized_rate)
    if method =='roll':
     rate_df = read_rate_df(raw_window) ## assuming that create_rate_df(days_window) has been run already
    else:
        rate_df = read_resampled_df(method)
    
    filtered_rate =  rate_df[(rate_df['date'] > startdate) & (rate_df['date'] < enddate)]
    edac_change = filtered_raw.drop_duplicates(subset='edac', keep='first', inplace=False) # Datetimes where the EDAC is increasing

    startdate_string= str(startdate).replace(" ", "_")
    startdate_string= startdate_string.replace(":", "")
    enddate_string= str(enddate).replace(" ", "_")
    enddate_string = enddate_string.replace(":", "")
    filtered_raw.loc[:, 'time_difference'] = filtered_raw['datetime'].diff().fillna(pd.Timedelta(seconds=0))
    filtered_raw.to_csv(path + 'events/rawEDAC_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False) # Save selected raw EDAC to file
    filtered_rate.to_csv(path + 'events/EDACrate_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)  # Save selected EDAc rate to file
    edac_change.to_csv(path + 'events/EDACchange'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)
    
    fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, figsize=(10,7))
    ax1.scatter(filtered_raw['datetime'],filtered_raw['edac'], label='Raw EDAC', s=3)
    ax2.plot(filtered_rate['date'], filtered_rate['daily_rate'], marker='o', label ='EDAC count rate')
    ax3.plot(normalized_rate['date'],normalized_rate['normalized_rate'],marker='o', label='Normalized EDAC count rate')
    ax3.set_xlabel('Date', fontsize = 12)
    ax1.set_ylabel('EDAC count', fontsize = 12)
    ax2.set_ylabel('EDAC count rate', fontsize = 12)
    ax3.set_ylabel('EDAC normalized count rate', fontsize=12)
    ax3.tick_params(axis='x', rotation=20)  # Adjust the rotation angle as needed
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    #plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    plt.tight_layout(pad=2.0)
    plt.savefig(path+'events/edac_'+startdate_string + '-' + enddate_string + '.png', dpi=300, transparent=False)
    plt.show()

def plot_rates_all(method): # Plots the EDAC count rate for the whole time period, before and after normalization
    standardized_df= read_standardized_rates(method)
    print(standardized_df)
    df_sun = process_sidc_ssn()
    start_date = datetime.strptime('2004-01-01',"%Y-%m-%d")
    index_exact =  np.where((df_sun['date']==start_date))[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed= savgol_filter(df_sun['daily_sunspotnumber'], savgolwindow_sunspots, 3)

    fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, figsize=(10,8))
    
    ax1.plot(standardized_df['date'], standardized_df['daily_rate'], label='Count rate')
    ax1.plot(standardized_df['date'],standardized_df['gcr_component'], label='Savitzky-Golay fit') 
    ax2.plot(standardized_df['date'],standardized_df['standardized_rate'], color ='#4daf4a',label='Standardized count rate')
    ax3.plot(df_sun['date'],df_sun['daily_sunspotnumber'], color='#f781bf',label="Sunspot number")
    ax3.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,
             color='#a65628',
             label='Smoothed sunspot number')
    ax3.set_ylabel('Sunspot number')
    ax3.set_xlabel('Date', fontsize = 10)
    ax1.set_ylabel('EDAC count rate [#/day]', fontsize = 10)
    ax2.set_ylabel('Standardized EDAC count rate [#/day]', fontsize = 10)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()


def skewed_gaussian_model(method):
    df = read_normalized_rates(method)
    df = df[df['date'] < pd.to_datetime('2017-09-01')]
    binsize = 0.1
    max_rate = np.max(df['normalized_rate'])
    min_rate = np.min(df['normalized_rate'])
    bins = np.arange(min_rate, max_rate+binsize, binsize) # Choose the size of the bins
    counts, bin_edges = np.histogram(df['normalized_rate'], bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df_hist = pd.DataFrame({'bin_center': bin_centers, 'count': counts})
    #test_df.to_csv(path + 'temp_test.txt', sep='\t', index=False) # Save to file

    model = SkewedGaussianModel()   # Create a SkewedGaussianModel
    params = model.guess(counts, x=bin_centers) # Guess initial parameters
    result = model.fit(counts, params, x=bin_centers) # Fit the model to the data

    #print(result.fit_report())
    fitted_params = result.params.valuesdict()
    df_model = pd.DataFrame({'bin_center' : bin_centers, 'model_fit':  result.best_fit})

    df_model['product'] = df_model['bin_center']*df_model['model_fit']
    sum_product_model = df_model['product'].sum()
    total_occurrences_model = df_model['model_fit'].sum()
    calculated_mean_model = sum_product_model/total_occurrences_model
    df_hist['product'] = df_hist['bin_center']*df_hist['count']
    sum_product_hist = df_hist['product'].sum()
    total_occurrences_hist = df_hist['count'].sum()
    calculated_mean_hist = sum_product_hist/total_occurrences_hist
    center = fitted_params['center']
    #df_test.to_csv(path + 'gaussian_test_new.txt', sep='\t', index=False) # Save to file
    # Plot the data and the fitted model
    plt.figure()
    #plt.plot(bin_centers, counts, label='Data')
    counts, bin_edges, _= plt.hist(df['normalized_rate'],bins=bins, density=False, ec='black', label='Distribution of normalized EDAC daily rates')
    plt.scatter(bin_centers, result.best_fit, label='Fitted model', color='#ff7f00')
    plt.axvline(x = center, color = 'red', label = 'center: ' +str(round(center,4)))
    plt.axvline(x=calculated_mean_hist, color='#4daf4a', label="calculated mean of the histogram, " +str(round(calculated_mean_hist,4)))
    plt.axvline(x=calculated_mean_model, color='#f781bf', label="calculated mean of the fitted model, " + str(round(calculated_mean_model,4)))
    plt.legend()
    plt.xlabel('Normalized EDAC daily rate')
    plt.ylabel('Occurrences')
    plt.title('Fitting with Skewed Gaussian Model')
    plt.show()

def plot_histogram_rates(method):
    df = read_normalized_rates(method)
    print("df: ", df)
    #df = df[df['date'] < pd.to_datetime('2022-05-01')]
    data = df['normalized_rate']
    #data = df['standardized_rate']
    binsize = 0.7
    max_rate = np.max(data)
    min_rate = np.min(data)
    bins = np.arange(min_rate, max_rate+binsize, binsize) # Choose the size of the bins
    #print(bins)
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    #print("bin_edges: ", bin_edges)
    plt.figure()
    plt.hist(data, bins=bin_edges, color='#4daf4a',edgecolor='black')
    plt.title('Normalized rate distribution')
    plt.xlabel('Count rate')
    plt.ylabel('Occurrences')
    plt.show()

def fit_skewnorm(method):
    df = read_normalized_rates(method)
    binsize = 0.5
    data = df['normalized_rate']
    max_rate = np.max(data)
    min_rate = np.min(data)
    bins = np.arange(min_rate, max_rate+binsize, binsize) # Choose the size of the bins
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    print(counts, bin_centers)
    print(sum(counts*bin_centers)/sum(counts))
    #a, loc, scale = 5.3, -0.1, 2.2
    #data = stats.skewnorm(a, loc, scale).rvs(1000)
    ae, loce, scalee = stats.skewnorm.fit(data)
    print(stats.skewnorm.fit(data))
    plt.figure()
    plt.hist(data, bins=bins,density = True, alpha=0.6, ec='black')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.skewnorm.pdf(x,ae, loce, scalee)#.rvs(100)
    plt.axvline(x=loce, label='Mean')
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()

def group_by_6_months(date):
    return pd.Timestamp(date.year, ((date.month-1)//6)*6 + 1, 1)



def eyeball_standardization_old():
 
    method = 'nonroll'
    df = read_normalized_rates(method)
    print(df)
    data_mean = df['standardized_rate'].mean()
    data_std_dev = df['standardized_rate'].std()
    print(data_mean, data_std_dev)
    upper_threshold = data_mean+2*data_std_dev
    thresholdcolor = '#984ea3'#'#e41a1c'
    lower_threshold = data_mean-1*data_std_dev
    start_date = datetime.strptime('2004-01-01',"%Y-%m-%d")
    suncolor = 'red' #'#4daf4a'
    df_sun = process_sidc_ssn()
    index_exact =  np.where((df_sun['date']==start_date))[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed= savgol_filter(df_sun['daily_sunspotnumber'], savgolwindow_sunspots, 3)

    fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, figsize=(10,8))
    
    ax1.plot(df['date'], df['daily_rate'], label='EDAC count rate')
    ax1.plot(df['date'],df['gcr_component'], label='Savitzky-Golay fit')

    ax2.plot(df['date'], df['standardized_rate'], 
             color='#4daf4a',
             label='Standardized count rate')
    ax2.axhline(upper_threshold, color= thresholdcolor, label='Threshold: ' + str(upper_threshold))
    #ax2.axhline(lower_threshold, color= thresholdcolor, label='Threshold: ' + str(lower_threshold))
    ax3.set_xlabel('Date', fontsize = 10)
    ax1.set_ylabel('Count rate [#/day]', fontsize = 10)
    ax2.set_ylabel('Standardized count rate [#/day]', fontsize = 10)
    ax3.plot(df_sun['date'],df_sun['daily_sunspotnumber'], color='#f781bf',label="Number of sunspots")
    ax3.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,
             color='#a65628',
             label='Smoothed sunspots')
    ax3.set_ylabel('Sunspot number')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    #plt.show()


    spike_df = df.copy()
    spike_df = spike_df[(spike_df['standardized_rate'] >= upper_threshold) | (spike_df['standardized_rate'] <= lower_threshold)]
    spike_df = spike_df[(spike_df['standardized_rate'] >= upper_threshold)]
    troughs= spike_df[spike_df['standardized_rate'] <= lower_threshold]
    peaks = spike_df[spike_df['standardized_rate']>= upper_threshold]
    print("troughs: ", troughs)
    print("peaks: ", peaks)
    spike_df.reset_index(inplace=True)
    #print("spike_df: ", spike_df)
    
    spike_df['6_month_group'] = spike_df['date'].apply(group_by_6_months)
    grouped_df = spike_df.groupby('6_month_group').size().reset_index()
    grouped_df.columns=['datebin','counts']
    grouped_df['datebin'] = grouped_df['datebin'] + pd.DateOffset(months=3)
    print(grouped_df)
    stormy_total = grouped_df['counts'].sum()

    print("Number of stormy days: ", stormy_total)
    spike_df.to_csv(path + 'datesoutsidethreshold.txt', sep='\t', index=False) # Save to file   
    fig, ax1 = plt.subplots(figsize=(10,6))
    spikecolor = '#377eb8'
    suncolor = '#a65628'
    ax1.plot(grouped_df['datebin'],grouped_df['counts'],
             marker='o',color=spikecolor,
             label='Number of days outside thresholds')
    ax2=ax1.twinx()
    #ax2.plot(df_sun['date'],df_sun['daily_sunspotnumber'], label="Number of sunspots")
    ax2.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,color=suncolor,
             label='Smoothed sunspots')
    sun_limit = max(sunspots_smoothed)+10
    ax2.set_ylim([0, max(sunspots_smoothed+10)])
    ax1.set_xlabel('Date', fontsize = 10)
    ax1.set_ylabel('Number of stormy days', fontsize = 10)
    ax2.set_ylabel('Sunspot number')
    ax1.tick_params(axis="y", labelcolor=spikecolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Sunspot number and number of stormy days in 6 month bins')
    plt.show()
def eyeball():
    method = 'nonroll'
    df = read_normalized_rates(method)
    
    upper_threshold = 2.177
    thresholdcolor = '#984ea3'#'#e41a1c'
    lower_threshold = -3#-1.324
    start_date = datetime.strptime('2004-01-01',"%Y-%m-%d")
    suncolor = 'red' #'#4daf4a'
    df_sun = process_sidc_ssn()
    index_exact =  np.where((df_sun['date']==start_date))[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed= savgol_filter(df_sun['daily_sunspotnumber'], savgolwindow_sunspots, 3)

    fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, figsize=(10,8))
    
    ax1.plot(df['date'], df['daily_rate'], label='EDAC count rate')
    ax1.plot(df['date'],df['gcr_component'], label='Savitzky-Golay fit')
    ax2.plot(df['date'], df['normalized_rate'], 
             color='#4daf4a',
             label='Normalized count rate')
    ax2.axhline(upper_threshold, color= thresholdcolor, label='Threshold: ' + str(upper_threshold))
    ax2.axhline(lower_threshold, color= thresholdcolor, label='Threshold: ' + str(lower_threshold))
    ax3.set_xlabel('Date', fontsize = 10)
    ax1.set_ylabel('Count rate [#/day]', fontsize = 10)
    ax2.set_ylabel('Standardized count rate [#/day]', fontsize = 10)
    ax3.plot(df_sun['date'],df_sun['daily_sunspotnumber'], color='#f781bf',label="Number of sunspots")
    ax3.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,
             color='#a65628',
             label='Smoothed sunspots')
    ax3.set_ylabel('Sunspot number')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    #plt.show()


    spike_df = df.copy()
    
    spike_df = spike_df[(spike_df['normalized_rate'] >= upper_threshold) | (spike_df['normalized_rate'] <= lower_threshold)]
    spike_df.reset_index(inplace=True)
 
    
    spike_df['6_month_group'] = spike_df['date'].apply(group_by_6_months)
    grouped_df = spike_df.groupby('6_month_group').size().reset_index()
    grouped_df.columns=['datebin','counts']
    grouped_df['datebin'] = grouped_df['datebin'] + pd.DateOffset(months=3)
    print(grouped_df)
    stormy_total = grouped_df['counts'].sum()

    print("Number of stormy days: ", stormy_total)
    spike_df.to_csv(path + 'datesoutsidethreshold.txt', sep='\t', index=False) # Save to file   
    fig, ax1 = plt.subplots(figsize=(10,6))
    spikecolor = '#377eb8'
    suncolor = '#a65628'
    ax1.plot(grouped_df['datebin'],grouped_df['counts'],
             marker='o',color=spikecolor,
             label='Number of days outside thresholds')
    ax2=ax1.twinx()
    #ax2.plot(df_sun['date'],df_sun['daily_sunspotnumber'], label="Number of sunspots")
    ax2.plot(df_sun['date'], sunspots_smoothed,
             linewidth=1,color=suncolor,
             label='Smoothed sunspots')
    sun_limit = max(sunspots_smoothed)+10
    ax2.set_ylim([0, max(sunspots_smoothed+10)])
    ax1.set_xlabel('Date', fontsize = 10)
    ax1.set_ylabel('Number of stormy days', fontsize = 10)
    ax2.set_ylabel('Sunspot number')
    ax1.tick_params(axis="y", labelcolor=spikecolor)
    ax2.tick_params(axis="y", labelcolor=suncolor)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Sunspot number and number of stormy days in 6 month bins')
    plt.show()


def process_sidc_ssn(): # returns sunspot dataframe
    column_names = ['year', 'month', 'day', 'date_fraction','daily_sunspotnumber','std','observations','status']
    df_sun = pd.read_csv(path+'SN_d_tot_V2.0.csv',names = column_names, sep=';')
    df_sun = df_sun[df_sun['daily_sunspotnumber']>=0]

    df_sun['date'] = pd.to_datetime(df_sun[['year', 'month', 'day']]) 
    df_sun = df_sun[['date', 'daily_sunspotnumber']]
    return df_sun

def fit_distribution_to_sine_rates():

    timewindow_start = pd.to_datetime("2017-09-01")
    timewindow_end = pd.to_datetime("2017-09-17")
    
    rate_df = gcr_edac_v2()
    
    rate_df['subtract'] = rate_df['daily_rate']-rate_df['sine_fit']
    rate_df['divide'] = rate_df['daily_rate']/rate_df['sine_fit']
    #rate_df =  rate_df[(rate_df['date'] > timewindow_start) & (rate_df['date'] < timewindow_end)]
    
    df_sun = pd.read_csv('files/SunSpots_2000-2020.txt',names=["date", "count"], skiprows=0, sep="\t", parse_dates = ['date'])
    #df_sun =  df_sun[(df_sun['date'] > timewindow_start) & (df_sun['date'] < timewindow_end)]
    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,6))
    ax1.plot(rate_df['date'],rate_df['daily_rate'], label='EDAC count rate')
    ax1.plot(rate_df['date'],rate_df['sine_fit'],label='sine fit')
    #ax1.plot(rate_df['date'],rate_df['subtract'],label='subtract')
    #ax1.plot(rate_df['date'],rate_df['divide'],label='divide')
    start_date = datetime.strptime('2004-01-01',"%Y-%m-%d")

    # Indices where the date is the same as the beginning window date
    index_exact =  np.where((df_sun['date']==start_date))[0][0]
    df_sun = df_sun.iloc[index_exact:]
    savgolwindow_sunspots = 601
    sunspots_smoothed= savgol_filter(df_sun['count'], savgolwindow_sunspots, 3)
    ax2.set(xlabel="Date", ylabel = "Number of sunspots", title = "Solar Cycle")
    ax1.set(ylabel= "EDAC count rate [#/day]")
    ax2.plot(df_sun['date'],df_sun['count'], label="Number of sunspots")
    ax2.plot(df_sun["date"], sunspots_smoothed,linewidth=1,label='Smoothed sunspots, savgolwindow = ' + str(savgolwindow_sunspots))
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.show()


    binsize = 0.03
    data = rate_df['sine_fit']
    max = np.max(data)
    min = np.min(data)
    bins = np.arange(min, max+binsize, binsize) # Choose the size of the bins
    plt.figure(figsize=(8,6))
    result =plt.hist(data,bins = bins, density = False, color='#FF6B6B',ec='black')
    for i in range(len(result[0])):
        plt.text(bins[i], result[0][i], str(int(result[0][i])), ha='left', va='bottom')
    plt.xlabel("EDAC count rate")
    plt.ylabel("Occurrences")
    plt.show()


    binsize = 0.03
    data = rate_df['sine_fit']
    max = np.max(data)
    min = np.min(data)
    bins = np.arange(min, max+binsize, binsize) # Choose the size of the bins
    plt.figure(figsize=(8,6))
    result =plt.hist(data,bins = bins, density = False, color='#FF6B6B',ec='black')
    for i in range(len(result[0])):
        plt.text(bins[i], result[0][i], str(int(result[0][i])), ha='left', va='bottom')
    plt.xlabel("EDAC count rate")
    plt.ylabel("Occurrences")
    plt.show()

from distfit import distfit
def try_distfit():
    method = 'nonroll'
    df = read_normalized_rates(method)

    print(df)
    X=df['daily_rate']
    #X = np.random.normal(0, 2, 1000)
    #y = [-8,-6,0,1,2,3,4,5,6]

    # Initialize model
    dfit = distfit()

    # Find best theoretical distribution for empirical data X
    dfit.fit_transform(X)
    dfit.plot()
    plt.show()

    # Make prediction
    #dfit.predict(y)
    #dfit.plot()
    
import obspy
from obspy.signal.detrend import polynomial
def try_obspy():
    tr = obspy.read()[0].filter("highpass", freq=2)
    tr.data += 6000 + 4 * tr.times() ** 2
    tr.data -= 0.1 * tr.times() ** 3 + 0.00001 * tr.times() ** 5
    #data = tr.data
    method = 'nonroll'
    df = read_normalized_rates(method)

    data=df['daily_rate']
    test = polynomial(data, order=12, plot=True)  
    
import statsmodels.api as sm
def try_statsmodels():
    method = 'nonroll'
    df = read_normalized_rates(method)


    # Assuming your data has a column named 'value' containing the time series values
    ts_values = df['daily_rate']

    # Apply the HP filter
    cycle, trend = sm.tsa.filters.hpfilter(ts_values)

    # Plot the original time series and the trend component
    plt.figure(figsize=(10, 6))
    plt.plot(ts_values, label='Original Time Series')
    plt.plot(trend, label='Trend Component from HP filter', linestyle='--')
    plt.plot(df['gcr_component'],label='Savitzky-Golay fit', color='#e41a1c')
    plt.legend()
    plt.title('Original Time Series and Trend Component')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
path = 'files/' # Path of where files are located
raw_edac_filename = 'MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt' # Insert the path of the raw EDAC file
patched_edac_filename = 'raw_edac/patched_mex_edac.txt'
raw_window = 5 # The time bin for calculating the rate for the EDAC curve to be normalized
smooth_window = 11 # The time bin for calculating the rate for the curve that is to be smoothed

def process_new_raw_edac(method): # Creates .txt files based on the raw EDAC. Do only once
    print("Method is: ", method)
    create_zero_set_correct(path+patched_edac_filename) # Create zeroset corrected EDAC file, needs to be done only once.
    zerosetcorrected_df = read_zero_set_correct() # Read the zeroset corrected file
    create_resampled_edac(zerosetcorrected_df, method) # Resample to a daily frequency. Needs to be done only once.
    if method == 'roll':
        create_rate_df(raw_window) # Creates daily rates based on resampled EDAC.
        create_rate_df(smooth_window) # Creates daily rates based on resampled EDAC
    #create_normalized_rates(method)
  
def main():
    method='nonroll'
    #process_new_raw_edac(method)
    #create_detrended_rates(method)
    #create_standardized_rates(method)
    #show_timerange(pd.to_datetime('2013-03-31 23:59:00'), pd.to_datetime('2013-04-23 00:00:00'), patched_edac_filename, method)
    
    plot_rates_all(method)
    #create_normalized_rates(method)
    #plot_histogram_rates(method)
   
    #create_standardized_rates(method)
    #eyeball_standardization()
    #eyeball()

    #try_distfit()
    #try_obspy()
    #try_statsmodels()
    print("End")

if __name__ == "__main__":
    main()


