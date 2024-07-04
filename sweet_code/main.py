from file_paths import get_data_dir
from plot_functions import plot_stormy_days_bin

if __name__ == "__main__":
    # process_raw_edac()
    # standardize()
    file_path = get_data_dir()
    plot_stormy_days_bin(file_path)
