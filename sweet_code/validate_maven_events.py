import pandas as pd
import os
from datetime import datetime, timedelta
from read_from_database import read_sep_events_maven
from detect_sw_events import read_sweet_event_dates
from parameters import SWEET_VALIDATION_DIR

def generate_search_area(start_date):
    """
    Time period where matches in literature
    are searched for
    """
    return [start_date.date() + timedelta(days=i) for i in range(-2, 3)]


def validate_lee_2017():
    sweet_df = read_sweet_event_dates()
    sweet_df['date'] = sweet_df['date'].dt.date
    maven_df = read_sep_events_maven()
    maven_df = maven_df[maven_df['source'].str.contains('Lee') & maven_df['source'].str.contains('2017')]
    maven_df  = maven_df[["onset_time", "source", "comment"]]
    #maven_dates = maven_df["onset_time"].tolist()
    maven_dict = dict.fromkeys(maven_df['onset_time'], [])
    for event_date in maven_dict.keys():
        maven_dict[event_date] = generate_search_area(event_date)
        print(event_date)
    results = {}
    for key, values in maven_dict.items():
        
        # For each SEP onset time
        # look for SWEET events in the time area -3 -3 days after
        # maven_event_type = values[0]
        
        maven_event_type = maven_df["comment"][maven_df["onset_time"] == key].values[0]
        print(maven_event_type)
        matching_indices = sweet_df[sweet_df['date'].isin(values)].index.tolist()
        if len(matching_indices) >0:
            match_found = True
            type_found = sweet_df["type"].iloc[matching_indices[0]]
            
        else:
            match_found=False
            type_found = "Missed"
        
        #match_found = any(date in sweet_df['date'].values for date in values)
        #type_found = sweet_df[sweet_df["date"]
        #                          == key.date()]["type"]
        results[key] = type_found, maven_event_type
    
    df = pd.DataFrame(list(results.items()),
                      columns=['onset_time', 'key'])
    df[["SWEET_found", "comment"]] = pd.DataFrame(df["key"].tolist(), index=df.index)
    df = df[["onset_time", "SWEET_found", "comment"]]
    file_name = "validate_lee_2017.txt"
    if not os.path.exists(SWEET_VALIDATION_DIR):
        os.makedirs(SWEET_VALIDATION_DIR )
    df.to_csv(SWEET_VALIDATION_DIR / file_name,
              sep='\t', index=False)
    print(f"File created: {SWEET_VALIDATION_DIR}/{file_name}")
    


def read_validation_lee_2017():
    file_name = "validate_lee_2017.txt"
    df = pd.read_csv(SWEET_VALIDATION_DIR / file_name,
    sep = '\t',
    parse_dates=['onset_time'])
    return df

def create_table_validation_lee_2017():
    df = read_validation_lee_2017()
    for index, row in df.iterrows():
        row_string = (
                f"{row['onset_time'].date()} & "
                f"{row['comment']} & "
                f"{row['SWEET_found']}\\\\"
            )
        print(row_string)

if __name__ == "__main__":
    
    #validate_lee_2017()
    df =read_validation_lee_2017()
    print(df)
    #create_table_validation_lee_2017()
    # create_table_validation_lee_2017()
    #sweet_df =read_sweet_event_dates()
    #sweet_df['date'] = sweet_df['date'].dt.date
    #print(sweet_df)
    #df_sep = df[df["type"]=="SEP"]
    #print(df_sep)
    #print(df["duration"].value_counts())
    #print(df)