from parameters import MAVEN_SEP_DIR
import cdflib
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv(dotenv_path='C:\\Users\\shayl\\OneDrive - NTNU\\edac_repo\\edac_work\\.env')

# from https://pds-ppi.igpp.ucla.edu/collection/urn:nasa:pds:maven.sep.calibrated:data.anc
cdf_file_1 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170910_v04_r09.cdf")
cdf_file_2 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170911_v04_r04.cdf")
cdf_file_3 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170912_v04_r04.cdf")
cdf_file_4 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170913_v04_r04.cdf")
cdf_file_5 = cdflib.CDF(MAVEN_SEP_DIR / "calibrated_spectra/mvn_sep_l2_s1-cal-svy-full_20170914_v04_r07.cdf")

variables = cdf_file_1.cdf_info().zVariables
print("Variables:", variables)
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