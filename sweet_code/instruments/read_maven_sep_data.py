import pandas as pd
from io import StringIO
from ..parameters import MAVEN_SEP_DIR, LOCAL_DIR



def process_maven_sep_hourly_data():
    path = MAVEN_SEP_DIR / "omniweb/"

    filename_base = "maven_f_flux_hr_" #2025"

    for i in range (2014,2026, 1):
        fmt_filename = filename_base + str(i) + '.fmt'
        data_filename = filename_base +str(i) + '.lst'

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
                            names=headers,
                            sep=r"\s+",)
            #print(df["YEAR"])
            df['date'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')
            df['datetime'] = df['date'] + pd.to_timedelta(df['Hour'], unit='h')

            df.to_csv(path / f'processed_maven_sep_{i}.txt', sep='\t', index=False)
            print(path / f'processed_maven_sep_{i}.txt created')


def create_merged_maven_sep_flux_data():
    """
     Merge all Solar Energetic Particle (SEP) Experiment: Ion and Electron hourly fluxes in 
     Forward look direction from s1 sensor for year 2014-2025.
     Each file contains one calendar year of data.
     The original data source has a resolution of 8 seconds.
    """
    path = MAVEN_SEP_DIR / "omniweb/"
    df = pd.read_csv(path / 'processed_maven_sep_2014.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])

    for i in range(2015, 2026, 1):
        add_on_df = pd.read_csv(path / f'processed_maven_sep_{i}.txt',
                     skiprows=0, sep="\t", parse_dates=['date'])

        df = pd.concat([df, add_on_df], axis=0, ignore_index=True)

    df = df.sort_values(by="datetime")
    df.to_csv(path / f'processed_maven_sep_all.txt', sep='\t', index=False)
    print(path / f'processed_maven_sep_all.txt created')

def read_maven_sep_flux_data():

    """
    Reads the merged processed data from the s1 sensor
    in the forward look direction on MAVEN/SEP
    """

    path = MAVEN_SEP_DIR / "omniweb/"
    df = pd.read_csv(path / 'processed_maven_sep_all.txt',
                     sep="\t", parse_dates=['date'])

    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

if __name__ == "__main__":
    # process_maven_sep_hourly_data()
    # create_merged_maven_sep_flux_data()
    df = read_maven_sep_flux_data()

    