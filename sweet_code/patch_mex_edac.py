import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from parameters import PROCESSED_DATA_DIR, RAW_DATA_DIR


def patch_update_13_11_2024():
    original_df = pd.read_csv(RAW_DATA_DIR / "patched_mex_edac_specialization_project.txt",
                        skiprows=0, sep="\t", parse_dates=['datetime'])
    new_df = read_update_13_11_2024()
    new_df = new_df[["datetime", "NDMW0D0G"]]
    new_df.columns = ["datetime", "edac"]
    new_df.replace({'edac': ' '}, np.nan, inplace=True)
    new_df = new_df.dropna(subset=['edac'])
    new_df = new_df.reset_index()
    new_df = new_df.drop(columns=['index'])

    new_df['edac'] = new_df['edac'].astype(int)
    df = pd.concat([original_df, new_df], axis=0, ignore_index=True)
    df.to_csv(PROCESSED_DATA_DIR / "patched_mex_edac.txt", sep='\t', index=False)
    print(f"File {PROCESSED_DATA_DIR}/ patched_mex_edac.txt created")


def read_update_13_11_2024():
    header_df = ["datetime",
                    "NACP1300",
                    "NACP1301",
                    "NACP2300",
                    "NACP2301",
                    "NACW0D0G",
                    "NDMW0D0G"]
    df = pd.read_csv(RAW_DATA_DIR / 
                         "NACP1300_NACP1301_NACP2300_NACP2301_NAC20D0G_NDMW0D0G.csv", 
                         skiprows=1,
                         sep=",")
    df.columns = header_df
    df['datetime']= pd.to_datetime(df['datetime'])
    return df



def plot_update_13_11_2024():
    
    fig, ax1 = plt.subplots(1, figsize=(10,7.5))
    edac_names =    ["NACP1300",
                    "NACP1301",
                    "NACP2300",
                    "NACP2301",
                    "NACW0D0G",
                    "NDMW0D0G"]
    for edac in edac_names:
        filename =  f'zerosetcorrected_{edac}.txt'
        df = pd.read_csv(PROCESSED_DATA_DIR / filename,
                     sep="\t", parse_dates = ["datetime"])  
        ax1.plot(df["datetime"], df["edac"], label= edac)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("EDAC counter [#]")
    
    ax1.tick_params(which="major", length=10, labelsize=10)
    ax1.tick_params(which="minor", length=6, labelsize=10)
    # x-axis ticks
    ax1.tick_params(axis="x", rotation=10)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))

    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(5))
    ax1.grid()
    ax1.legend()
    fig.suptitle("Zeroset-corrected MEX EDACs", fontsize=16)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_update_13_11_2024_individual():
    """
    Investigate how the data looks like
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / "zerosetcorrected_NDMW0D0G.txt",
                     sep="\t", parse_dates = ["datetime"])
    print(df)

    fig, ax1 = plt.subplots(1, figsize=(10,7.5))
    ax1.plot(df["datetime"], df["edac"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("EDAC counter [#]")
    

    ax1.tick_params(which="major", length=10, labelsize=10)
    ax1.tick_params(which="minor", length=6, labelsize=10)
    # x-axis ticks
    ax1.tick_params(axis="x", rotation=10)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))

    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(2))
    ax1.grid()
    fig.suptitle("Zeroset-corrected NDMW0D0G", fontsize=16)
    plt.tight_layout(pad=1.0)
    plt.show()


def zeroset_correct_edacs_13_11_2024():
    """
    Create the zero-set corrected dataframe of the raw EDAC counters
    """

    print("--------- Starting the zeroset correction ---------")
    pre_df = read_update_13_11_2024()
    for k in range(1, len(pre_df.columns)):
        
        df = pre_df[["datetime", pre_df.columns[k]]]
        print("Processing ", pre_df.columns[k])
        df.columns = ["datetime", "edac"]
        df['edac'].replace(' ', np.nan, inplace=True)
        df.dropna(subset=['edac'], inplace=True)
        df.reset_index(inplace=True)
        df['edac'] = df['edac'].astype(int)

        diffs = df.edac.diff()
        # Finding the indices where the EDAC counter decreases
        indices = np.where(diffs < 0)[0]
        print("This EDAC data set was zero-set ", len(indices), " times.")
        for i in range(0, len(indices)):
            prev_value = df.loc[[indices[i]-1]].values[-1][-1]
            if i == len(indices)-1:  # The last time the EDAC counter goes to zero
                df.loc[indices[i]:, 'edac'] = \
                    df.loc[indices[i]:, 'edac'] + prev_value
            else:
                df.loc[indices[i]:indices[i+1]-1, 'edac'] = \
                    df.loc[indices[i]:indices[i+1]-1, 'edac'] + prev_value
                
        filename = "zerosetcorrected_" + str(pre_df.columns[k]) + ".txt"
        df.to_csv(PROCESSED_DATA_DIR / filename, sep='\t', index=False)
        print(f"File {PROCESSED_DATA_DIR}/{filename} created")
    

if __name__ == "__main__":
    # plot_update_13_11_2024()
    patch_update_13_11_2024()
    # plot_update_13_11_2024()
    # plot()