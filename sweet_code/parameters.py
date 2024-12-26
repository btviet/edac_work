from typing import Final

from dotenv import load_dotenv
from file_paths import get_data_dir

load_dotenv()

UPPER_THRESHOLD: Final = 2.5
LOWER_THRESHOLD: Final = 0
FD_NUMBER_DAYS: Final = 3
SUNSPOTS_SAVGOL = 601
RATE_SAVGOL = 1095
POLYORDER_SAVGOL = 3

LOCAL_DIR = get_data_dir()
RAW_DATA_DIR = LOCAL_DIR / "raw_data/"
PROCESSED_DATA_DIR = LOCAL_DIR / "processed/"
SWEET_EVENTS_DIR = LOCAL_DIR / "edac_events/"
OLD_SWEET_DIR = LOCAL_DIR / "old_sweet/"
SEP_VALIDATION_DIR = LOCAL_DIR / "sep_validation/"
CME_VALIDATION_DIR = LOCAL_DIR / "cme_validation/"
DATABASE_DIR = LOCAL_DIR / "database/"
FORBUSH_VALIDATION_DIR = LOCAL_DIR / "forbush_validation/"
SWEET_VALIDATION_DIR = LOCAL_DIR / "sweet_validation"
TOOLS_OUTPUT_DIR = LOCAL_DIR / "tools_output/"
MAVEN_SEP_DIR = LOCAL_DIR / "maven_sep_data/"
MEX_ASPERA_DIR = LOCAL_DIR / "mex_aspera_data/"

# colors from https://personal.sron.nl/~pault/#sec:qualitative
RAW_EDAC_COLOR = "#0077BB"
ZEROSET_COLOR = "#E55F3F"
RATE_EDAC_COLOR = "#33BBEE"
RATE_FIT_COLOR = "#CC6677"
DETRENDED_EDAC_COLOR = "#009988"
STANDARDIZED_EDAC_COLOR = "#EE7733"
THRESHOLD_COLOR = "#EE3377"
SSN_COLOR = "#EE7733"
SSN_SMOOTHED_COLOR = "#000000"
BINNED_COLOR = "#999933"
BRAT_GREEN = "#8ACE00"
FONTSIZE_AXES_LABELS = 16
FONTSIZE_TITLE = 18
FONTSIZE_AXES_TICKS = 14
FONTSIZE_LEGENDS = 12
DETREND_METHOD = 'division'
