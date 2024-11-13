import pandas as pd

# Code to extract the dates from Table in Guo et al. (2018).

with open('first_side_fd.txt') as f:
    raw_data = f.readlines()
    stripped = raw_data[0].split(' ')
    print(stripped)
    df_old = pd.DataFrame(columns=['num',
                                   'onset_time',
                                   'nadir_time',
                                   'dose_rate_before_onset',
                                   'nadir_dose_rate',
                                   'drop_msl',
                                   'flux_before_onset',
                                   'nadir_flux',
                                   'drop_maven'])
    df = pd.DataFrame(columns=['num', 'onset_time'])
    for i in range(1, 122, 1):
        print(i)
        current_index = stripped.index(str(i))
        if i <= 120:
            next_index = stripped.index(str(i+1))
        else:
            next_index = -1
        current_row = stripped[current_index:next_index]
        print(current_row)
        onset_time = pd.to_datetime(f"{current_row[1]} {current_row[2]}")
        print("datetime_obj: ", onset_time)
        new_row = pd.DataFrame({'num': [i], 'onset_time': [onset_time]})

    df = pd.concat([df, new_row], ignore_index=True)
    df['onset_time'] = df['onset_time'].dt.strftime('%d/%m/%Y %H:%M:%S')
    """
    df = df["onset_time"]
    print(df)
    df.to_csv("msl_rad_fds.txt",
              sep="\t",
              index=False,
              )
    """
