from parameters import MAVEN_SEP_DIR
import cdflib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
load_dotenv(dotenv_path='C:\\Users\\shayl\\OneDrive - NTNU\\edac_repo\\edac_work\\.env')

# from https://pds-ppi.igpp.ucla.edu/collection/urn:nasa:pds:maven.sep.calibrated:data.anc




def create_plots():
    cdf_file_1 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170910_v04_r09.cdf")
    cdf_file_2 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170911_v04_r04.cdf")
    cdf_file_3 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170912_v04_r04.cdf")
    cdf_file_4 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170913_v04_r04.cdf")
    cdf_file_5 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170914_v04_r07.cdf")


    fto_a = cdf_file_1.varget('a_fto_rates')
    timestamps = cdf_file_1.varget('epoch')
    timestamps_2 = cdf_file_2.varget('epoch')
    timestamps_3 = cdf_file_3.varget('epoch')
    timestamps_4 = cdf_file_4.varget('epoch')
    timestamps_5 = cdf_file_5.varget('epoch')

    # first_column = electron_energy[:,0]
    utc_times = [cdflib.cdfepoch.to_datetime(tt) for tt in timestamps]
    utc_times_2 = [cdflib.cdfepoch.to_datetime(tt) for tt in timestamps_2]
    utc_times_3 = [cdflib.cdfepoch.to_datetime(tt) for tt in timestamps_3]
    utc_times_4 = [cdflib.cdfepoch.to_datetime(tt) for tt in timestamps_4]
    utc_times_5 = [cdflib.cdfepoch.to_datetime(tt) for tt in timestamps_5]
    utc_times = utc_times + utc_times_2 + utc_times_3 + utc_times_4 + utc_times_5
    
    variables = cdf_file_1.cdf_info().zVariables
    print("Variables:", variables)
    for i in range(4, len(variables)-1):
        
        plt.figure(figsize=(8,6))
        variable_1 = cdf_file_1.varget(variables[i])
        variable_2 = cdf_file_2.varget(variables[i])
        variable_3 = cdf_file_3.varget(variables[i])
        variable_4 = cdf_file_4.varget(variables[i])
        variable_5 = cdf_file_5.varget(variables[i])
        variable =  np.concatenate((variable_1, variable_2, variable_3, variable_4, variable_5))
        plt.plot(utc_times, variable)
        # plt.plot(utc_times, cdf_file.varget(variables[i]))
        # print(np.shape(cdf_file.varget(variables[i]))#
        plt.title(variables[i])
        plt.xlabel('Date')
        plt.savefig(MAVEN_SEP_DIR / 
                    f'calibrated_spectra/{variables[i]}.png',
                    dpi=300, transparent=False)
        plt.close()



def attempt():
    cdf_file_1 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170910_v04_r09.cdf")
    cdf_file_2 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170911_v04_r04.cdf")
    cdf_file_3 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170912_v04_r04.cdf")
    cdf_file_4 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170913_v04_r04.cdf")
    cdf_file_5 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170914_v04_r07.cdf")


    variables = cdf_file_1.cdf_info().zVariables
    print("Variables:", variables)
    # print(variables[1])
    # print(cdf_file_2.varget('r_ion_flux_unc'))
    data_1 = cdf_file_1.varget('f_ion_flux')
    print(data_1)
    data_2 = cdf_file_2.varget('f_ion_flux')
    data_3 = cdf_file_3.varget('f_ion_flux')
    data_4 = cdf_file_4.varget('f_ion_flux')
    data_5 = cdf_file_5.varget('f_ion_flux')
    data =  np.concatenate((data_1, data_2, data_3, data_4, data_5))
    energy_channels =cdf_file_1.varget('f_ion_energy')
    #print("energy channels: ", np.shape(energy_channels))
    #print(np.shape(data))
    #print(type(data))
    timestamps_1 = cdf_file_1.varget('epoch')
    utc_times_1 = [cdflib.cdfepoch.to_datetime(tt) for tt in timestamps_1]

    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    sns.heatmap(data, cmap='plasma', cbar=True)
    plt.title('Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    #plt.savefig(MAVEN_SEP_DIR / f'spectra_plots/{variables[i]}.png',
    #            dpi=300, transparent=False)
    plt.show()
    
    # plt.close()


def test_sns():
    import pandas as pd
    data = np.random.random((12462, 28))  # Example random data
# Convert to DataFrame for labeling (optional)
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    sns.heatmap(df, cmap='viridis', cbar=True)
    plt.title('Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def read_maven_sep_flux_data(filename):
    """
    Reads files from
    https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi
    """
    path = MAVEN_SEP_DIR / "omniweb/"
    fmt_filename = filename + '.fmt'
    data_filename = filename + '.lst'

    fmt_file = path / fmt_filename
    data_file = path / data_filename

    with open(fmt_file, 'r') as f:
        # Skipping the rows before the relevant info
        lines = f.readlines()[4:]
    headers = []
    for line in lines:
        elements = line.strip().split()
        if "I" in elements[-1:][0]:
                # Year, DOY, HOUR are integers
            headers.append(elements[1])
        elif len(elements) == 5:
                energy_range = "".join(elements[1:-2])
                unit = elements[-2:-1][0]
                item_header = str(energy_range) + " " + str(unit)
                headers.append(item_header)
        elif len(elements) == 4:
                item_header = elements[1] + " " + elements[-2]
                headers.append(item_header)  
        else:
            raise Exception("MAVEN/SEP format error")
    with open(data_file, 'r') as f:
        data = f.read()
    df = pd.read_csv(StringIO(data), 
                     delim_whitespace=True,
                     names=headers)

    df['date'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')
    df['datetime'] = df['date'] + pd.to_timedelta(df['Hour'], unit='h')
    return df




def read_maven_sep_flux_data_old():
    """
    files from omniweb:
    https://omniweb.gsfc.nasa.gov/ftpbrowser/flux_spectr_m.html
    """
    path = MAVEN_SEP_DIR / "omniweb/"
    fmt_file = path / 'maven_f_flux_hourly_sept_2017.fmt' 
    with open(fmt_file, 'r') as f:
        lines = f.readlines()[4:]
    headers = []
    for line in lines:
        """
        Assume ions and electrons are not plotted in the same figure
        """
        elements = line.strip().split()
        # item_index = elements[0]
        # item_format = elements[:-1]

        if len(elements[1:-1]) >= 3:
            energy_range = "".join(elements[1:-2])
            unit = elements[-2:-1][0]

            item_header = str(energy_range) + " " + str(unit)
            headers.append(item_header)
        else:
            headers.append(elements[1:-1][0])
 
    print(headers)

    """
    # df = pd.read_csv(StringIO(data), delim_whitespace=True)

    # Reshape the DataFrame to have energy channels as rows
    heatmap_data = df.iloc[:, 3:].T  # Transpose so channels (1-21) are rows
    heatmap_data.columns = [f"Day {i}" for i in range(1, heatmap_data.shape[1] + 1)]  # Rename columns
    heatmap_data.index = [f"Channel {i}" for i in range(1, heatmap_data.shape[0] + 1)]  # Rename rows

    plt.figure(figsize=(12, 6))  # Adjust the size as needed
    sns.heatmap(heatmap_data, cmap='plasma', cbar=True, robust=True)
    plt.title("Energy Channels vs. Days Heatmap")
    plt.xlabel("Days")
    plt.ylabel("Energy Channels")
    plt.show()
    """

if __name__ == "__main__":
    # attempt()
    # test_sns()
    # test_2()
    # read_maven_sep_flux_data()
    filename = 'maven_f_flux_hourly_sept_2017'
    read_maven_sep_flux_data(filename)