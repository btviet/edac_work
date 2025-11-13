def plot_count_rate_with_fit():
    import pandas as pd
    
    
    
    df = read_resampled_df()
    detrended_df = read_detrended_rates()
    rate_mean = round(df['daily_rate'].mean(), 3)
    print(df.sort_values(by='daily_rate'))

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["daily_rate"],
             label=f'MEX EDAC daily rate, mean = {rate_mean} counts per day',
             color=RATE_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("MEX EDAC count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax1.plot(
        detrended_df["date"],
        detrended_df["gcr_component"],
        label="Savitzky-Golay fit",
        color=RATE_FIT_COLOR 
    )


    major_y_locator = MultipleLocator(2)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax1.xaxis.set_minor_locator(YearLocator(1))
    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)

    ax1.grid()
    ax1.set_ylim([-1, 20])
    #ax2.set_ylim([0, 18])
    # fig.suptitle("brat")
    fig.suptitle("MEX EDAC count rate",
                 fontsize=FONTSIZE_TITLE-4)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_gcr_fit_ssn():
    detrended_df = read_detrended_rates()
    df_sun = process_sidc_ssn()
    print(detrended_df)
    start_date = datetime.strptime("2004-01-01", "%Y-%m-%d")
    index_exact = np.where(df_sun["date"] == start_date)[0][0]
    df_sun = df_sun.iloc[index_exact:]
    # sunspots_smoothed = savgol_filter(df_sun["daily_sunspotnumber"],
    # SUNSPOTS_SAVGOL, 3)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax2.plot(
        detrended_df["date"],
        detrended_df["gcr_component"],
        label="MEX EDAC GCR component",
        color=RAW_EDAC_COLOR
    )

    ax1.plot(
        df_sun["date"],
        df_sun["daily_sunspotnumber"],
        label="Sunspot number",
        color=SSN_COLOR
    )

    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Sunspot number [#]", fontsize=FONTSIZE_AXES_LABELS, color=SSN_COLOR)
    ax2.set_ylabel("Count rate [#/day]", fontsize=FONTSIZE_AXES_LABELS, color=RAW_EDAC_COLOR)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(axis="y", labelcolor=SSN_COLOR) 
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS, 
                    labelcolor=RAW_EDAC_COLOR)
    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(which='minor', length=6)
    major_y_locator = MultipleLocator(50)
    ax1.yaxis.set_major_locator(major_y_locator)

    minor_y_locator = MultipleLocator(25)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    ax1.grid()
    plt.suptitle("EDAC GCR component and the sunspot no. between Jan. 2004 and Jul. 2024",
                 fontsize=FONTSIZE_TITLE)
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="upper left",
        fontsize=FONTSIZE_LEGENDS, bbox_to_anchor=(0.02, 1)) 
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    ax2.set_ylim([detrended_df["gcr_component"].min()-0.05, 2.1])
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=1.0)
    plt.show()


def plot_variable_noise_threshold():
    df = read_detrended_rates()

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df["date"], df["detrended_rate"],
             label='Detrended count rate',
             color=DETRENDED_EDAC_COLOR,  # BRAT_GREEN,
             linewidth=2)
    ax1.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Detrended count rate [#/day]",
                   fontsize=FONTSIZE_AXES_LABELS)
    ax1.plot(
        df["date"],
        df["gcr_component"]+1,
        label="Savitzky-Golay fit",
        color=RATE_FIT_COLOR 
    )
    # ax1.axhline(UPPER_THRESHOLD, color='black',
    #            linewidth=2,
    #            linestyle='dashed',
    #            label=f'Threshold of {UPPER_THRESHOLD}')

    major_x_locator = YearLocator(4)
    ax1.xaxis.set_major_locator(major_x_locator)
    ax1.minorticks_on()
    minor_x_locator = YearLocator(1)
    ax1.xaxis.set_minor_locator(minor_x_locator)

    major_y_locator = MultipleLocator(2)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(1)
    ax1.yaxis.set_minor_locator(minor_y_locator)

    ax1.tick_params(which='minor', length=6)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax1.legend(fontsize=FONTSIZE_LEGENDS)
    ax1.grid()
    # fig.suptitle("brat")
    fig.suptitle("MEX EDAC detrended count rate",
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout(pad=1.0)
    plt.show()   


def process_sidc_ssn():
    """
    Processes the SSN data from SIDC
    Parameters
    ----------
    file_path : Path
        The location of the file containing the SSN data

    Returns
    -------
    pd.DataFrame
        Dataframe containing the sunspot number
        for each date
    """
    column_names = [
        "year",
        "month",
        "day",
        "date_fraction",
        "daily_sunspotnumber",
        "std",
        "observations",
        "status",
    ]
    df_sun = pd.read_csv(RAW_DATA_DIR / "SN_d_tot_V2.0.csv",
                         names=column_names, sep=";")
    df_sun = df_sun[df_sun["daily_sunspotnumber"] >= 0]

    df_sun["date"] = pd.to_datetime(df_sun[["year", "month", "day"]])
    df_sun = df_sun[["date", "daily_sunspotnumber"]]
    return df_sun
