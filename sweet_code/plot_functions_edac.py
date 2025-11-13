
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.dates import YearLocator, DayLocator
from matplotlib.ticker import MultipleLocator
from edac_work.sweet_code.process_edac.processing_edac import read_raw_edac_2025, read_rawedac

from parameters import (
    IMA_COLOR,
    BRAT_GREEN,
    DETRENDED_EDAC_COLOR,
    FONTSIZE_AXES_LABELS,
    FONTSIZE_AXES_TICKS,
    FONTSIZE_LEGENDS,
    FONTSIZE_TITLE,
    LOCAL_DIR,
    RATE_EDAC_COLOR,
    RATE_FIT_COLOR,
    RAW_DATA_DIR,
    RAW_EDAC_COLOR,
    SEP_VALIDATION_DIR,
    SSN_COLOR,
    SSN_SMOOTHED_COLOR,
    STANDARDIZED_EDAC_COLOR,
    SUNSPOTS_SAVGOL,
    SWEET_EVENTS_DIR,
    THRESHOLD_COLOR,
    UPPER_THRESHOLD,
    ZEROSET_COLOR,
    RAD_E_COLOR,
    RAD_B_COLOR,
    TOOLS_OUTPUT_DIR
)

def plot_raw_edac():

    df = read_raw_edac_2025()
    df_2= read_rawedac()
    print(df_2)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["datetime"], df["edac"],
             label='New MEX EDAC',
             linewidth=2)
    ax1.plot(df_2["datetime"], df_2["edac"], label='Old MEX EDAC data', linewidth=2, linestyle='dashed')
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count [#]", fontsize=FONTSIZE_AXES_LABELS)
    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(2)
    ax1.xaxis.set_minor_locator(minor_x_locator)
    major_y_locator = MultipleLocator(5000)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(2500)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend()
    ax1.grid()
    fig.suptitle("brat")

    plt.tight_layout(pad=1.0)
    plt.show()

if __name__ == "__main__":
    plot_raw_edac()