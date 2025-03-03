
import pandas as pd
from parameters import EUHFORIA_DIR
# Read a DSV file with a specified delimiter (e.g., `|`)


filename = 'euhforia_Mars.dsv',
def read_euhforia_mars_dsv_file(filename):
    df = pd.read_csv(EUHFORIA_DIR / filename, delim_whitespace=True,
    parse_dates=['date'])
    return df

