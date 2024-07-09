from detect_sw_events import detect_edac_events
from plot_functions import (
    create_fd_plots,
    create_sep_plots,
    create_stormy_plots,
    plot_real_eruption_dates,
    plot_real_sep_onsets,
    plot_stormy_days_bin,
)
from processing_edac import process_raw_edac
from standardize_edac import standardize
from validate_cme_events import validate_cme_eruptions
from validate_sep_events import validate_sep_onsets


def sweet():
    process_raw_edac()
    standardize()
    detect_edac_events()


def sweet_events_plots():
    create_stormy_plots()
    create_sep_plots()
    create_fd_plots()
    plot_stormy_days_bin()


def validate_real_events():
    validate_cme_eruptions()
    validate_sep_onsets
    plot_real_eruption_dates()
    plot_real_sep_onsets()


if __name__ == "__main__":
    sweet()
    sweet_events_plots()
    validate_real_events()
