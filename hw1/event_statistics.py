import time
import pandas as pd
import numpy as np

def read_csv(filepath):
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    Event count is defined as the number of events recorded for a given patient.
    '''
    all_counts = events['patient_id'].value_counts()
    alive_counts = all_counts.drop(list(mortality['patient_id']))
    dead_counts = all_counts[list(mortality['patient_id'])]
    avg_dead_event_count = dead_counts.mean()
    max_dead_event_count = dead_counts.max()
    min_dead_event_count = dead_counts.min()
    avg_alive_event_count = alive_counts.mean()
    max_alive_event_count = alive_counts.max()
    min_alive_event_count = alive_counts.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    dates = events[['patient_id','timestamp']].groupby(['patient_id']).timestamp.nunique()
    alive_dates = dates.drop(list(mortality['patient_id']))
    dead_dates = dates[list(mortality['patient_id'])]
    avg_dead_encounter_count = dead_dates.mean()
    max_dead_encounter_count = dead_dates.max()
    min_dead_encounter_count = dead_dates.min()
    avg_alive_encounter_count = alive_dates.mean()
    max_alive_encounter_count = alive_dates.max()
    min_alive_encounter_count = alive_dates.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    length = (events[['patient_id','timestamp']].groupby('patient_id').max() - events[['patient_id','timestamp']].groupby('patient_id').min())['timestamp']
    alive_dates = length.drop(list(mortality['patient_id'])).dt.days
    dead_dates = length[list(mortality['patient_id'])].dt.days
    avg_dead_rec_len = dead_dates.mean()
    max_dead_rec_len = dead_dates.max()
    min_dead_rec_len = dead_dates.min()
    avg_alive_rec_len = alive_dates.mean()
    max_alive_rec_len = alive_dates.max()
    min_alive_rec_len = alive_dates.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    train_path = '../data/train/'

    # ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
