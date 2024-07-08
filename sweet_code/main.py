from detect_sw_events import detect_edac_events
from plot_functions import (
    create_fd_plots,
    create_sep_plots,
    create_stormy_plots,
    plot_real_eruption_dates,
)
from processing_edac import process_raw_edac
from standardize_edac import standardize
from validate_events import validate_cme_eruptions


def sweet():
    process_raw_edac()
    standardize()
    detect_edac_events()


def sweet_events_plots():
    create_stormy_plots()
    create_sep_plots()
    create_fd_plots()


def main():
    sweet()
    sweet_events_plots()
    validate_cme_eruptions()
    plot_real_eruption_dates()


if __name__ == "__main__":
    # main()
    # validate_cme_eruptions()
    # detect_real_sep()
    # read_cme_events()
    # combine_findings_sep_fd()
    # analyze_validation_results()
    # validate_cme_eruptions()
    plot_real_eruption_dates()
