from typing import Final

from dotenv import load_dotenv
from file_paths import get_data_dir

load_dotenv()

UPPER_THRESHOLD: Final = 2
LOWER_THRESHOLD: Final = 0
FD_NUMBER_DAYS: Final = 3
SUNSPOTS_SAVGOL = 601
RATE_SAVGOL = 500

LOCAL_DIR = get_data_dir()
RAW_DATA_DIR = LOCAL_DIR / "raw_data/"
PROCESSED_DATA_DIR = LOCAL_DIR / "processed/"
SWEET_EVENTS_DIR = LOCAL_DIR / "edac_events/"
OLD_SWEET_DIR = LOCAL_DIR / "old_sweet/"
# colors from https://personal.sron.nl/~pault/#sec:qualitative
RAW_EDAC_COLOR = "#0077BB"
RATE_EDAC_COLOR = "#33BBEE"
RATE_FIT_COLOR = "CC6677"
DETRENDED_EDAC_COLOR = "#009988"
STANDARDIZED_EDAC_COLOR = "#EE7733"
THRESHOLD_COLOR = "#EE3377"
SSN_COLOR = "#EE7733"
SSN_SMOOTHED_COLOR = "#CC3311"
BINNED_COLOR = "#999933"
