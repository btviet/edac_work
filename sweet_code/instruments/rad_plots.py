import sys
import os

parent_directory = os.path.abspath('../edac_work')
sys.path.append(parent_directory)

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import YearLocator

from sweet_code.instruments.read_msl_rad_data import read_msl_rad_doses
from sweet_code.parameters import (RAD_B_COLOR,
                                   RAD_E_COLOR,
                                   FONTSIZE_LEGENDS,
                                   FONTSIZE_AXES_LABELS, 
                                   FONTSIZE_AXES_TICKS,
                                   FONTSIZE_TITLE)


def plot_msl_rad_all():
    df = read_msl_rad_doses()
    print(df)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df['datetime'], df['B_dose'], label='B dose rate',
             color=RAD_B_COLOR)
    ax1.plot(df['datetime'], df['E_dose'], label='E dose rate', linestyle='dashed',
             color=RAD_E_COLOR)
    
    ax1.xaxis.set_major_locator(YearLocator(2))
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator())
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylim(-1, 2)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()
    fig.suptitle("MSL/RAD dose rates from August 2012 to July 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.close()
    #plt.show()


    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 7))
    ax1.plot(df['datetime'], df['B_dose'], label='B dose rate',
             color=RAD_B_COLOR)
    ax1.plot(df['datetime'], df['E_dose'], label='E dose rate', linestyle='dashed',
             color=RAD_E_COLOR)
    
    ax1.xaxis.set_major_locator(YearLocator(2))
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator())
    ax1.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    # ax1.set_ylim(-1, 2)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()

    ax2.plot(df['datetime'], df['B_dose'], label='B dose rate',
             color=RAD_B_COLOR)
    ax2.plot(df['datetime'], df['E_dose'], label='E dose rate', linestyle='dashed',
             color=RAD_E_COLOR)
    
    ax2.xaxis.set_major_locator(YearLocator(2))
    ax2.xaxis.set_minor_locator(YearLocator(1))
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)

    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.set_ylabel("Dose rate [mGy/day]", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax2.set_ylim(-0.01, 1.1)
    ax1.set_ylim(0, 9)
    ax2.legend(fontsize=FONTSIZE_LEGENDS)
    ax2.grid()
    lower_xlim = df['datetime'].iloc[0] - pd.Timedelta(days=180)
    higher_xlim = df['datetime'].iloc[-1] + pd.Timedelta(days=180)
    ax2.set_xlim(lower_xlim, higher_xlim)


    fig.suptitle("MSL/RAD dose rates from August 2012 to July 2024",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    
    plt.subplots_adjust(hspace=0.2)   
    #plt.savefig('msl_rad_all_v4.eps',
    #            dpi=300, transparent=False,
    #             bbox_inches="tight", pad_inches=0.05)
    #plt.savefig('msl_rad_all_v4.png',
    #            dpi=300, transparent=False,
    #             bbox_inches="tight", pad_inches=0.05)
    
    plt.show()


if __name__ == "__main__":
    print("wo")
    plot_msl_rad_all()