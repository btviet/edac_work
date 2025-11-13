import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import matplotlib as mpl

#from matplotlib.gridspec import GridSpec
#from pandas.plotting import register_matplotlib_converters
#from sqlalchemy import all_
#register_matplotlib_converters()

from edac_work.sweet_code.process_edac.processing_edac import (
    read_rawedac,
    read_zero_set_correct,
    read_resampled_df,
    read_rolling_rates,
    read_zero_set_correct,
)

from datetime import datetime, timedelta
import matplotlib.dates as mdates
import statistics
import scipy.optimize
#from scipy.signal import savgol_filter
#import os
#import sys
import time
import scipy.stats
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rcParams
default_font = rcParams['font.family']
print("Default font:", default_font)
rcParams['font.sans-serif'] = ['Myriad', 'Arial', 'DejaVu Sans']
rcParams['font.family'] = 'sans-serif'
default_font = rcParams['font.family']
print("Default font:", default_font)

from parameters import LOCAL_DIR

#rcParams['font.sans-serif'] = ['Myriad Pro']  # Specify Myriad Pro or Myriad Sans Serif
from matplotlib import font_manager
#font_path = 'C:/Users/shayl/Downloads/myriad-pro.otf'
#custom_font = font_manager.FontProperties(fname=font_path)
#default_tick_width = rcParams["xtick.major.width"]
#print("Default tick width:", default_tick_width)

path="files/"
patched_edac_filename = 'patched_mex_edac.txt'


def read_patched_rawedac(patched_edac_path): # Reads the patched MEX EDAC
    df = pd.read_csv(patched_edac_path,skiprows=0, sep="\t",parse_dates = ['datetime'])
    return df

def create_rawedac_df(): # Creates a dataframe from the raw data provided by MEX
    path = 'files/' # The location where the EDAC files are
    #df = pd.read_csv(path+ 'MEX_NDMW0D0G_2024_03_18_19_12_06.135.txt', skiprows=15, sep="\t", parse_dates=['# DATE TIME'])
    df = read_patched_rawedac(path+patched_edac_filename)
    df2 = pd.read_csv(path+'MEX_NDMW0D0G_MEX_NACW0D0G_2024_04_11_07_37_22.272.txt',skiprows=16, sep="\t",parse_dates=['# DATE TIME'])
    columns2 = list(df2.columns.values)
    df2.rename(columns={columns2[0]: 'datetime', columns2[1]: 'edac', columns2[2]: 'edac_nac'}, inplace=True) 
    #df = df.dropna(subset=['edac'])
    df2 = df2[df2['edac'] != ' ']
    df2['edac'] = df2['edac'].astype(int)
    df2 = df2.drop('edac_nac', ax2is=1)

    combined_df = pd.concat([df, df2])

    combined_df.drop_duplicates(subset='datetime', keep='first', inplace=True)

    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.sort_values(by="datetime")
    combined_df.set_index('datetime')
    combined_df.to_csv(path + 'createrawedac.txt', sep='\t', index=False) # Save to file    
    return combined_df

def create_zero_set_correct_v0(): # Returns the zero-set corrected dataframe of the raw EDAC counter
    start_time = time.time()
    print("--------- Starting the zeroset correction ---------")
    ### df = read_rawedac_df(raw_edac_file) ## For not patched EDACs

    df =create_rawedac_df()
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

def read_zero_set_correct_v0():
    df = pd.read_csv(
        path+'zerosetcorrected_edac.txt',skiprows=0, sep="\t",parse_dates = ['datetime'])
    return df

def new_rates():
    df = read_zero_set_correct()
    #df=  df[df['datetime'] > pd.to_datetime('2024-03-16 00:00:00')]
    df= df.set_index('datetime') 

    last_df = df.resample('D').last().ffill()
    df_resampled = df.resample('D').first().ffill()
    df_resampled.reset_index(inplace=True)
    last_df.reset_index(inplace=True)
    df_resampled['edac_last'] = last_df['edac']
    df_resampled.rename(columns={'datetime': 'date', 'edac':'edac_first'}, inplace=True)
    df_resampled['daily_rate'] = df_resampled['edac_last']-df_resampled['edac_first']
    
    df_resampled['date'] = df_resampled['date']+pd.Timedelta(hours=12)
    df_resampled.to_csv(path + 'new_rates.txt', sep='\t', index=False) # Save to file 
    return df_resampled

def fit_sin(tt, yy): # Helping function for gcr_edac()
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax2(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "max2cov": np.max(pcov), "rawres": (guess,popt,pcov)}

def gcr_edac():
    #rate_df = read_rate_df(smooth_window)
    rate_df = new_rates()

    start_date = rate_df['date'].iloc[0]
    tt = np.array([(x - start_date).days for x in rate_df['date']])
    yy = rate_df['daily_rate']
    res = fit_sin(tt, yy)
    x_datetime = np.array([start_date + pd.Timedelta(days=x) for x in tt])
    df = pd.DataFrame({'date':x_datetime,'daily_rate':  res["fitfunc"](tt)})
    return df

def create_normalized_rates(): # Return the normalized EDAC rate
    gcr_component = gcr_edac()
    first_gcr = gcr_component['date'].iloc[0]
    last_gcr = gcr_component['date'].iloc[-1]
    rate_df = new_rates()
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
    normalized_df['gcr_component'] = gcr_component['daily_rate']
    normalized_df['normalized_rate'] = normalized_df['daily_rate']/normalized_df['gcr_component']
    
    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,6))
    
    ax1.plot(normalized_df['date'],normalized_df['daily_rate'], label='EDAC daily rate')
    ax1.plot(normalized_df['date'],normalized_df['gcr_component'], label='GCR component of EDAC daily rate')
    ax2.plot(normalized_df['date'], normalized_df['normalized_rate'], label='Normalized daily EDAC rate')
    #ax1.set_ylim([-0.5, 7])
    #ax2.set_ylim([-0.5, 7])
    #plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    ax2.set_xlabel('Date', fontsize = 12)
    ax1.set_ylabel('EDAC daily rate', fontsize = 12)
    ax2.set_ylabel('EDAC normalized daily rate', fontsize = 12)
    ax2.tick_params(axis='x', rotation=20)  # Adjust the rotation angle as needed
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.tight_layout(pad=0.5)
    #plt.savefig(path+'events/edac_'+startdate_string + '-' + enddate_string + '.png', dpi=300, transparent=False)
    plt.show()
    normalized_df.to_csv(path + 'normalized_edac_test.txt', sep='\t', index=False) # Save to file

#create_normalized_rates()
def read_normalized_rates():
    df = pd.read_csv(path + 'normalized_edac_test.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    return df

def read_new_rates():
    df = pd.read_csv(
        path+'new_rates.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    return df

def read_rawedac_df():
    df = pd.read_csv(
        path+'createrawedac.txt',skiprows=0, sep="\t",parse_dates = ['datetime'])
    return df

def show_timerange(startdate, enddate, raw_edac_file):
    startdate_string= str(startdate).replace(" ", "_")
    startdate_string= startdate_string.replace(":", "")
    enddate_string= str(enddate).replace(" ", "_")
    enddate_string = enddate_string.replace(":", "")
    #raw_edac =  read_patched_rawedac(raw_edac_file)
    raw_edac = read_rawedac()
    # raw_edac = read_rawedac_df()#create_rawedac_df()

    zeroset_edac = read_zero_set_correct()
    rate_df  = read_resampled_df()
    # rate_df = read_new_rates()
    filtered_raw = raw_edac.copy()
    filtered_raw =  filtered_raw[(filtered_raw['datetime'] > startdate) & (filtered_raw['datetime'] < enddate)]
    
    filtered_zeroset = zeroset_edac.copy()
    filtered_zeroset = filtered_zeroset[(filtered_zeroset['datetime'] > startdate) & (filtered_zeroset['datetime'] < enddate)]
    filtered_rate =  rate_df[(rate_df['date'] > startdate) & (rate_df['date'] < enddate)]

    #normalized_rate = read_normalized_rates()
    #normalized_rate =normalized_rate[(normalized_rate['date'] > startdate) & (normalized_rate['date'] < enddate)]
    
    edac_change = filtered_raw.drop_duplicates(subset='edac', keep='first', inplace=False) # Datetimes where the EDAC is increasing

    filtered_raw.loc[:, 'time_difference'] = filtered_raw['datetime'].diff().fillna(pd.Timedelta(seconds=0))
    #filtered_raw.to_csv(path + 'events/rawEDAC_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False) # Save selected raw EDAC to file
    #filtered_rate.to_csv(path + 'events/EDACrate_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)  # Save selected EDAc rate to file
    #edac_change.to_csv(path + 'events/EDACchange'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)

    raw_edac_color = '#004E00' #'#377eb8'
    rate_edac_color = '#ff7f0e' #'#ff7f00'
    rover_color = 'black' # "¨'#358DCC' 
    marker_color='#ff7f0e'
 
    fig, ax1 = plt.subplots(figsize=(3.55, 1.59))
    ax1.plot(filtered_raw['datetime'],filtered_raw['edac'], label='EDAC count', 
             #marker='o', markersize=1, markerfacecolor=raw_edac_color, markeredgecolor=raw_edac_color,
             linewidth=1, color=raw_edac_color)
    ax2 = ax1.twinx()  
    ax2.plot(filtered_rate['date'], filtered_rate['daily_rate'], label ='EDAC count rate',
              marker='o', markersize=2,  markerfacecolor=marker_color,markeredgecolor=marker_color,
              linewidth=1,
              color='darkgrey')
    plt.axvline(x=pd.to_datetime('2024-03-18 06:47:00'), 
        color=rover_color, linestyle='--', label='Time of rover \nobservation',
        linewidth=1)
    ax1.set_ylabel('Counts (#)', fontsize = 8, color=raw_edac_color)
    # ax1.set_xlabel('Date', fontsize = 12)
    ax2.set_ylabel('Count rate (#/day)', fontsize = 8, color=rate_edac_color)
    # Set the label color
    ax1.tick_params(axis="y", labelcolor=raw_edac_color)
    ax2.tick_params(axis="y", labelcolor=rate_edac_color)
    #ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=True)   
    ax1.tick_params(bottom=True, top=True, left=True, right=False)
    date_format = mdates.DateFormatter('%b %d')
    ax1.xaxis.set_major_formatter(date_format)

    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
   

    lower_xlim = min(filtered_rate['date'].iloc[0], filtered_raw['datetime'].iloc[0])
    higher_xlim = max(filtered_rate['date'].iloc[-1], filtered_raw['datetime'].iloc[-1])
    major_ticks_locations =  [pd.to_datetime('2024-03-18 00:00:00') + pd.Timedelta(days=7 * i) for i in range(-3, 4)]  # One week away, up to five weeks
    lower_xlim = lower_xlim - timedelta(days=1)
    higher_xlim = higher_xlim + timedelta(days=2)
    ax1.set_xlim(lower_xlim, higher_xlim)
    ax1.set_xticks(major_ticks_locations)
    ax1.tick_params(axis='x', direction='in', labelsize=8, labelbottom=True, labeltop=False)
    ax2.tick_params(axis='both', direction='in', labelsize=8)
    ax1.tick_params(axis='both', direction='in', labelsize=8)
    ax1.tick_params(axis='x', rotation=20, length=8)

    length_of_minor_ticks = 3
    width_of_minor_ticks = 0.5
    length_of_major_ticks = 5
    width_of_major_ticks=0.8
    ax1.tick_params(axis='x', which='minor', direction='in', bottom=True, top=True, 
                    width=width_of_minor_ticks,
                    length=length_of_minor_ticks)
    ax1.tick_params(axis='y', which='minor', direction='in', left=True, right=False, 
                    width=width_of_minor_ticks,
                    length=length_of_minor_ticks)
    ax1.tick_params(axis='y', which='major', direction='in', left=True, right=False, length=length_of_major_ticks,
                    width=width_of_major_ticks)

    ax2.tick_params(axis='y', which='minor', direction='in', left=False, right=True,
                    length=length_of_minor_ticks,
                    width=width_of_minor_ticks)
    
    ax2.tick_params(axis='y', which='major', direction='in', left=False, right=True, length=length_of_major_ticks,
                    width=width_of_major_ticks)
    ax1.tick_params(axis='x', which='major', direction='in', bottom=True, top=True, length=length_of_major_ticks,
                    width=width_of_major_ticks)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    axbox = ax1.get_position()
    ax1.legend(handles, labels, loc=(axbox.x0-0.06, axbox.y1-0.23),fontsize=6,
    handlelength=1.5)
    ## x more negative: <---
    ## x more positve: ---->
    ## y more negative: downwards
    #ax1.legend()
    #ax2.legend()
    
    
    #plt.suptitle('MEX EDAC counter during March 18th 2024 solar storm', fontsize=16)
    plt.tight_layout(pad=1.0)
    plt.savefig("ex_3.pdf", dpi=1200, transparent=True, bbox_inches='tight')
    plt.show()
    
    #fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,5))
    
    ##ax1.scatter(filtered_raw['datetime'],filtered_raw['edac'], label='Raw EDAC', s=3)
    #ax1.plot(filtered_raw['datetime'],filtered_raw['edac'], label='Raw EDAC')
    ##ax2.plot(filtered_zeroset['date'], filtered_zeroset['edac'], label ='Zeroset-corrected EDAC')
    #ax2.plot(filtered_rate['date'], filtered_rate['daily_rate'], label ='Daily rate')
    ##ax2.plot(normalized_rate['date'], normalized_rate['normalized_rate'], label='Normalized EDAC rate')
    ##plt.suptitle('December 5th, 2006 SEP event', fontsize=16)
    #ax2.set_xlabel('Date', fontsize = 14)
    #ax1.set_ylabel('EDAC count', fontsize = 14)
    #ax2.set_ylabel('EDAC daily rate', fontsize = 14)
    ##ax2.set_ylim(0, 5)  # Adjust the values as needed
    #ax2.tick_params(axis='x', rotation=20)  # Adjust the rotation angle as needed
    #ax1.grid()
    #ax2.grid()
    #ax1.legend()
    #ax2.legend()
    #plt.tight_layout(pad=2.0)
    #plt.savefig(path+'events/edac_'+startdate_string + '-' + enddate_string + '.png', dpi=300, transparent=False)
    #plt.show()


def plot_paper_figure(startdate, enddate):
    length_of_minor_ticks = 3
    width_of_minor_ticks = 0.5
    length_of_major_ticks = 5
    width_of_major_ticks=0.8
    tick_labels_fontsize=8
    label_fontsize=8
    # set this as True or False depending on if you want exported as svg:
    savefig = True
    # Folder where you save the data
    # <-Update this line for your system!->
    file_folder = LOCAL_DIR

    # load the dataset:
    filename = "SEP2_2024Mar15to2024Mar21.npz"
    sep_data = np.load(os.path.join(file_folder, filename), allow_pickle=True)

    # Pull the columns:
    epoch = sep_data["epoch_utc"]
    elec_energy = sep_data["electron_energy_keV"]
    elec_flux = sep_data["electron_flux_cm2sterskeV"]
    ion_energy = sep_data["ion_energy_keV"]
    ion_flux = sep_data["ion_flux_cm2sterskeV"]

    # Calculate the energy flux:
    eflux_unit = "$keV/cm^2/s/sr/keV$"
    elec_eflux = elec_energy[np.newaxis, :] * elec_flux
    ion_eflux = ion_energy[np.newaxis, :] * ion_flux

    norm = LogNorm(vmin=1, vmax=1e5)
    cmap = 'plasma'


    # mpl.rcParams['font.size'] = 8   # Font size for Nature
    fig = plt.figure(figsize=(7.25, 2.64)) # 4.94

    # Create a GridSpec with 2 rows and 2 columns (equal widths)
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[1, 1])
    fig.subplots_adjust(left=0.055, right=0.92, hspace=0.08)
    # Left figure spanning both rows
    ax11 = fig.add_subplot(gs[:, 0])  

    # Two stacked figures on the right
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])


    startdate_string= str(startdate).replace(" ", "_")
    startdate_string= startdate_string.replace(":", "")
    enddate_string= str(enddate).replace(" ", "_")
    enddate_string = enddate_string.replace(":", "")
    #raw_edac =  read_patched_rawedac(raw_edac_file)
    raw_edac = read_rawedac()
    # raw_edac = read_rawedac_df()#create_rawedac_df()

    zeroset_edac = read_zero_set_correct()
    rate_df  = read_resampled_df()
    # rate_df = read_new_rates()
    filtered_raw = raw_edac.copy()
    filtered_raw =  filtered_raw[(filtered_raw['datetime'] > startdate) & (filtered_raw['datetime'] < enddate)]
    
    filtered_zeroset = zeroset_edac.copy()
    filtered_zeroset = filtered_zeroset[(filtered_zeroset['datetime'] > startdate) & (filtered_zeroset['datetime'] < enddate)]
    filtered_rate =  rate_df[(rate_df['date'] > startdate) & (rate_df['date'] < enddate)]

    #normalized_rate = read_normalized_rates()
    #normalized_rate =normalized_rate[(normalized_rate['date'] > startdate) & (normalized_rate['date'] < enddate)]
    
    edac_change = filtered_raw.drop_duplicates(subset='edac', keep='first', inplace=False) # Datetimes where the EDAC is increasing

    filtered_raw.loc[:, 'time_difference'] = filtered_raw['datetime'].diff().fillna(pd.Timedelta(seconds=0))
    #filtered_raw.to_csv(path + 'events/rawEDAC_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False) # Save selected raw EDAC to file
    #filtered_rate.to_csv(path + 'events/EDACrate_'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)  # Save selected EDAc rate to file
    #edac_change.to_csv(path + 'events/EDACchange'+startdate_string + '-' + enddate_string + '.txt', sep='\t', index=False)

    raw_edac_color = '#004E00' #'#377eb8'
    rate_edac_color = '#ff7f0e' #'#ff7f00'
    rover_color = 'black' # "¨'#358DCC' 
    marker_color='#ff7f0e'
    # ax1.plot(filtered_raw['datetime'],filtered_raw['edac'], label='EDAC count', 
             #marker='o', markersize=1, markerfacecolor=raw_edac_color, markeredgecolor=raw_edac_color,
    #         linewidth=1, color=raw_edac_color)
    # ax11 = ax1.twinx()  
    ax11.plot(filtered_rate['date'], filtered_rate['daily_rate'], label ='EDAC count rate',
              marker='o', markersize=2,
              markerfacecolor=marker_color,
              markeredgecolor=marker_color,
              linewidth=1,
              color='darkgrey')
    ax11.axvline(x=pd.to_datetime('2024-03-18 06:47:00'), 
        color=rover_color, linestyle='--', label='Time of rover \nobservation',
        linewidth=1)
    #ax1.set_ylabel('Counts (#)', fontsize = 8, color=raw_edac_color)
    # ax1.set_xlabel('Date', fontsize = 12)
    ax11.set_ylabel('Count rate (#/day)', fontsize = label_fontsize)
    # Set the label color
    #ax1.tick_params(axis="y", labelcolor=raw_edac_color)
    ax11.tick_params(axis="y")
    #ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=True)   
    #ax1.tick_params(bottom=True, top=True, left=True, right=False)
    date_format = mdates.DateFormatter('%b %d')
    ax11.xaxis.set_major_formatter(date_format)

    ax11.xaxis.set_minor_locator(mdates.DayLocator())
    #ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax11.yaxis.set_minor_locator(AutoMinorLocator())
   

    lower_xlim = min(filtered_rate['date'].iloc[0], filtered_raw['datetime'].iloc[0])
    higher_xlim = max(filtered_rate['date'].iloc[-1], filtered_raw['datetime'].iloc[-1])
    major_ticks_locations =  [pd.to_datetime('2024-03-18 00:00:00') + pd.Timedelta(days=7 * i) for i in range(-3, 4)]  # One week away, up to five weeks
    lower_xlim = lower_xlim - timedelta(days=1)
    higher_xlim = higher_xlim + timedelta(days=2)
    ax11.set_xlim(lower_xlim, higher_xlim)
    ax11.set_xticks(major_ticks_locations)
    #ax1.tick_params(axis='x', direction='in', labelsize=8, labelbottom=True, labeltop=False)
    ax11.tick_params(axis='both', direction='in', labelsize=tick_labels_fontsize)
    #ax1.tick_params(axis='both', direction='in', labelsize=8)
    ax11.tick_params(axis='x', rotation=20)


    ax11.tick_params(axis='x', which='minor', direction='in', bottom=True, top=False, 
                    width=width_of_minor_ticks,
                    length=length_of_minor_ticks)
    #ax1.tick_params(axis='y', which='minor', direction='in', left=True, right=False, 
    #                width=width_of_minor_ticks,
    #                length=length_of_minor_ticks)
    #ax1.tick_params(axis='y', which='major', direction='in', left=True, right=False, length=length_of_major_ticks,
    #                width=width_of_major_ticks)

    ax11.tick_params(axis='y', which='minor', direction='in', left=True, right=False,
                    length=length_of_minor_ticks,
                    width=width_of_minor_ticks)
    
    ax11.tick_params(axis='y', which='major', direction='in', left=True, right=False, length=length_of_major_ticks,
                    width=width_of_major_ticks)
    ax11.tick_params(axis='x', which='major', direction='in', bottom=True, top=False, length=length_of_major_ticks,
                    width=width_of_major_ticks)

    #handles1, labels1 = ax1.get_legend_handles_labels()
    #handles2, labels2 = ax11.get_legend_handles_labels()
    #handles = handles1 + handles2
    #labels = labels1 + labels2
    #axbox = ax11.get_position()
    ##ax11.legend(handles2, labels2, loc=(axbox.x0-0.015, axbox.y1-0.03),fontsize=6,
    ##handlelength=1.5)

    ## x more negative: <---
    ## x more positve: ---->
    ## y more negative: downwards

    ax2.axvline(x=pd.to_datetime('2024-03-18 06:47:00'), 
        color=rover_color, linestyle='--', label='Time of rover \nobservation',
        linewidth=1)
    
    ax3.axvline(x=pd.to_datetime('2024-03-18 06:47:00'), 
        color=rover_color, linestyle='--', label='Time of rover \nobservation',
        linewidth=1)
    
    
    plt.setp(ax2.get_xticklabels(), visible=False)
    p = ax2.pcolormesh(
        epoch, ion_energy, ion_eflux.T, norm=norm,
        cmap=cmap, rasterized=True)
    p = ax3.pcolormesh(
    epoch, elec_energy, elec_eflux.T, norm=norm,
    cmap=cmap, rasterized=True)
    
    p = ax3.pcolormesh(
        epoch, elec_energy, elec_eflux.T, norm=norm,
        cmap=cmap, rasterized=True)

    # axis labeling

    # Y-ax2es are for energy, so need to be logarithmic:
    
    ax2.set_ylabel("Ion energy (keV)", fontsize=label_fontsize, labelpad=0)
    ax2.set_yscale('log')
    ax3.set_ylabel("Electron energy (keV)", fontsize=label_fontsize, labelpad=0)
    ax3.set_yscale('log')


    # Set y-ticks to appear at specific log intervals
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
                                
    ax3.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))

    ax3.tick_params(axis='x', labelsize=tick_labels_fontsize) 
    ax3.tick_params(axis='y', labelsize=tick_labels_fontsize, pad=0)
    ax2.tick_params(axis='y', labelsize=tick_labels_fontsize, pad=0)

    # X axis constrained and formatted:
    start_dt = datetime(2024, 3, 15)
    end_dt = datetime(2024, 3, 22)
    ax3.set_xlim([start_dt, end_dt])
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax3.tick_params(axis='x', rotation=20,
                    length=length_of_major_ticks,
                    width=width_of_major_ticks, pad=0
                    )
    
    ax2.tick_params(axis='y', which='major', length=length_of_major_ticks,
                    width=width_of_major_ticks)
    ax3.tick_params(axis='y', which='major', length=length_of_major_ticks,
                    width=width_of_major_ticks)
    ax2.tick_params(axis='y', which='minor', length=length_of_minor_ticks,
                    width=width_of_minor_ticks)
    ax3.tick_params(axis='y', which='minor', length=length_of_minor_ticks,
                    width=width_of_minor_ticks)

    # Make a shared colorbar:
    upper_bbox = ax2.get_position()
    lower_bbox = ax3.get_position() # ax2[-1].get_position()
    cax = fig.add_axes(
        [lower_bbox.x1 + 0.01, lower_bbox.y0, 0.01,
        (upper_bbox.y1 - lower_bbox.y0)])
    cbar = fig.colorbar(p, cax=cax)

    cbar.set_label(r"Flux ($\mathrm{keV/cm^2/s/sr/keV}$)", fontsize=label_fontsize, labelpad=1) 
    cbar.set_ticks(ticks=[1, 10, 1e2, 1e3, 1e4, 1e5])
    cbar.ax.tick_params(labelsize=label_fontsize, which='major', pad=1, length=length_of_major_ticks,
                    width=width_of_major_ticks) 

    cbar.ax.tick_params(which='minor', length=length_of_minor_ticks,
                    width=width_of_minor_ticks) 

    if savefig:
        save_name = "maven_edac_figure.svg"
        plt.savefig(os.path.join(file_folder, save_name), dpi=1200)
    else:
        plt.show()


def test():
    fig = plt.figure(figsize=(10, 5))

    # Create a GridSpec with 2 rows and 2 columns
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[3, 1])

    # Left figure spanning both rows
    ax1 = fig.add_subplot(gs[:, 0])  

    # Two stacked figures on the right
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    # Example plots
    ax1.plot([1, 2, 3], [4, 5, 6])
    ax2.plot([1, 2, 3], [6, 5, 4])
    ax3.plot([1, 2, 3], [7, 8, 9])

    #plt.tight_layout()
    plt.show()

import matplotlib.gridspec as gridspec
def test2():

    fig = plt.figure(figsize=(10, 5))

    # Create a GridSpec with 2 rows and 2 columns (equal widths)
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[1, 1])

    # Left figure spanning both rows
    ax1 = fig.add_subplot(gs[:, 0])  

    # Two stacked figures on the right
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    # Example plots
    ax1.plot([1, 2, 3], [4, 5, 6])
    ax2.plot([1, 2, 3], [6, 5, 4])
    ax3.plot([1, 2, 3], [7, 8, 9])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # show_timerange(pd.to_datetime('2024-02-20 23:59:00'), pd.to_datetime('2024-04-10 23:59:00'), path+patched_edac_filename)
    plot_paper_figure(pd.to_datetime('2024-02-20 23:59:00'), pd.to_datetime('2024-04-10 23:59:00'))
    # test2()