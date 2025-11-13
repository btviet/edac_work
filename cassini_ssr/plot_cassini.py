from process_mimi import read_cassini_mimi
from process_cassini_ssr import read_cassini_ssr
from process_satrad import read_satrad_5m
from process_ephemeris import read_ephemeris
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DayLocator
from matplotlib.ticker import MultipleLocator
from datetime import datetime

FONTSIZE_AXES_LABELS = 16
FONTSIZE_AXES_TICKS = 14
FONTSIZE_TITLE = 18


mimi_p8_color = '#0072B2'
ssr_a_sbe_color = '#009E73'
ssr_b_sbe_color = '#D55E00'
satrad_color = '#CC79A7'


def plot_summary():
    df_mimi = read_cassini_mimi()
    df_ssr = read_cassini_ssr()
    df_mimi['tP8'] = df_mimi['tP8'].astype(float)
    print(df_ssr)
    print(df_mimi)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True,
                                        figsize=(10, 6))
    ax1.plot(df_mimi['datetime'], df_mimi['tP8'], color=mimi_p8_color
             )
    ax2.plot(df_ssr['SCET_UTC'], df_ssr['SSR-A-SBE'], color=ssr_a_sbe_color,
              label='SBE')
    ax2.plot(df_ssr['SCET_UTC'], df_ssr['SSR-A-DBE']*20, color='black',
             label='DBE*20')

    ax3.plot(df_ssr['SCET_UTC'], df_ssr['SSR-B-SBE'], color=ssr_b_sbe_color,
             label='SBE')
    ax3.plot(df_ssr['SCET_UTC'], df_ssr['SSR-B-DBE']*20, color='black',
             label='DBE*20')

    ax1.set_ylim(0, 0.03)
    ax2.set_ylim(0, 4000)
    ax3.set_ylim(0, 4000)

    ax1.xaxis.set_major_locator(YearLocator(2))
    ax1.xaxis.set_minor_locator(YearLocator())

        # minor_y_locator = MultipleLocator(2500)
    ax2.yaxis.set_minor_locator(MultipleLocator(1000))
    ax3.yaxis.set_minor_locator(MultipleLocator(1000))


    ax1.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax1.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax2.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='minor', length=6, labelsize=FONTSIZE_AXES_TICKS)
    ax3.tick_params(which='major', length=10, labelsize=FONTSIZE_AXES_TICKS)


    ax1.set_ylabel("MIMI P8 Flux ", fontsize=FONTSIZE_AXES_LABELS)
    # ax1.set_ylabel(r"$1/\mathrm{KeV} \mathrm{cm}^2  \mathrm{ster} \mathrm{sec} $")
    # 1/(keV cm^2 sr s
    ax2.set_ylabel("SSR-A", fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_ylabel("SSR-B", fontsize=FONTSIZE_AXES_LABELS)
    ax3.set_xlabel("Date", fontsize=FONTSIZE_AXES_LABELS)
    fig.suptitle("Cassini SSR and MIMI P8 data",
                 fontsize=FONTSIZE_TITLE, y=0.95)

    ax2.legend()
    ax3.legend()
    ax1.grid()
    ax2.grid()
    ax3.grid()

    plt.show()


def plot_closeup():
    ""
    ""

    df_mimi = read_cassini_mimi()
    df_ssr = read_cassini_ssr()
    df_mimi['tP8'] = df_mimi['tP8'].astype(float)
    df_satrad = read_satrad_5m()
    df_satrad['(H+@40MeV)'] = df_satrad['(H+@40MeV)']/1000000
    current_date = datetime.strptime("2016-01-01", "%Y-%m-%d")
    #start_date = current_date - pd.Timedelta(hours=12)
    start_date = datetime.strptime("2016-01-01 07:00:00", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime("2016-01-01 13:00:00", "%Y-%m-%d %H:%M:%S")
    # end_date = current_date + pd.Timedelta(hours=12)
    #start_date = datetime.strptime("2015-04-15", "%Y-%m-%d")
    #end_date = datetime.strptime("2023-07-17", "%Y-%m-%d")
    df_ssr = df_ssr[(df_ssr['SCET_UTC']>=start_date) & (df_ssr['SCET_UTC'] <= end_date)]
    df_mimi = df_mimi[(df_mimi['datetime']>=start_date) & (df_mimi['datetime'] <= end_date)]
    df_satrad = df_satrad[(df_satrad['SCET_UTC']>=start_date) & (df_satrad['SCET_UTC'] <= end_date)]
    print(df_mimi)
    print(df_satrad)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df_ssr['SCET_UTC'], df_ssr['SSR-B-SBE'], color=ssr_b_sbe_color,
              label='SBE')
    
    mimi_ax= ax1.twinx()
    mimi_ax.plot(df_mimi['datetime'], df_mimi['tP8'], color=mimi_p8_color, label='MIMI P8')
    #mimi_ax.plot(df_satrad['SCET_UTC'], df_satrad['(H+@40MeV)'], color=satrad_color,
    #              label='SATRAD')
    
    #mimi_ax.set_ylim(0, 0.05)

    ax1.set_ylabel('SSR-A counts', fontsize=FONTSIZE_AXES_LABELS)
    mimi_ax.set_ylabel("MIMI P8 Flux", fontsize=FONTSIZE_AXES_LABELS)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = mimi_ax.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, loc='upper left')

    plt.show()


def plot_errors_shell():
    df_ssr = read_cassini_ssr()
    df_mimi = read_cassini_mimi()
    #df_mimi.sort_values(by=[])
    #print(df_ssr)


def plot_ephemeris():
    df = read_ephemeris()
    print(df)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['datetime'], df['scdist'])
    ax1.set_yscale('log')
    plt.show()
if __name__ == "__main__":
    # plot_summary()
    #plot_closeup()
    # plot_errors_shell()
    plot_ephemeris()