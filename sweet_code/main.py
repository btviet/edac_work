from detrend_edac import detrend
from plot_functions import (
    create_event_plots,
    create_fd_plots,
    create_msl_rad_dates_sep_plots,
    create_sep_plots,
    create_stormy_plots,
    plot_stormy_days_bin,
    plot_sweet_events_binned,
)
from processing_edac import process_raw_edac
from validate_sep_events import validate_sep_onsets
from validate_sweet_events import cross_check_sweet


def sweet():
    process_raw_edac()  # Reads raw EDAC, zerosets, resamples
    detrend()
    # detect_sweet_events() # Find SWEET SEP events and Fds


def rolling_method():
    # calculate_rolling_window_rate(5)
    # calculate_rolling_window_rate(11)
    detrend()


def plots():
    create_stormy_plots()  # Create plots of all SWEET stormy days
    create_event_plots()  # Create plots of SWEET SEP and FD events
    create_sep_plots()  # Create plots of SWEET SEP events
    create_fd_plots()  # Create plots of all SWEET FD events
    plot_stormy_days_bin()  # Plot number of SWEET stormy days in bins
    plot_sweet_events_binned()  # Plot number of SWEET event sin bins
    create_msl_rad_dates_sep_plots()  # Plot MSL/RAD SEP events
    # Plot all events in literature database


def validate_real_events():
    # validate_cme_eruptions()
    validate_sep_onsets()
    # plot_real_eruption_dates()
    # plot_real_sep_onsets()


def validate_sweet_events():
    cross_check_sweet()


if __name__ == "__main__":
    # rolling_method()
    sweet()
    # detrend()
    # detect_sweet_events() # Find SWEET SEP events and Fds
    # plots()
    # validate_real_events()
    # find_sep()
