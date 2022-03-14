import utils
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn import preprocessing

def read_csv(filepath):
    
    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map

def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    1. Create list of patients alive 
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    '''
    alive_dates = events[['patient_id','timestamp']].loc[~events['patient_id'].isin(mortality['patient_id'])].groupby(['patient_id']).max().reset_index()
    dead_dates = mortality[['patient_id','timestamp']]
    dead_dates['timestamp']= dead_dates['timestamp'].apply(pd.to_datetime) - timedelta(days = 30)
    
    indx_date = pd.concat([alive_dates,dead_dates]).reset_index(drop=True)
    indx_date.columns = ['patient_id','indx_date']
    indx_date.to_csv(deliverables_path+'etl_index_dates.csv',columns = ['patient_id','indx_date'],index = False)        
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    '''
    events_merge = pd.merge(events,indx_date, on = ['patient_id'])
    events_merge[['timestamp','indx_date']] = events_merge[['timestamp','indx_date']].apply(pd.to_datetime)
    filtered_events = events_merge[(events_merge['timestamp'] <= events_merge['indx_date']) & (events_merge['timestamp'] >= (events_merge['indx_date']-timedelta(days = 2000)))]
    filtered_events = filtered_events[['patient_id','event_id','value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv',index = False)    
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    '''
    #create column to indicate aggregation group
    filtered_events_df['event_type'] = filtered_events_df['event_id'].apply(lambda x: x[0])
    #replace events with event index
    idx_events = pd.merge(filtered_events_df,feature_map_df, on = 'event_id')
    #remove events with na values
    idx_events = idx_events[idx_events['value'].notna()]
    #DIAG/DRUG sum group
    d_events = idx_events[idx_events['event_type']=='D']
    d_events = d_events.groupby(['patient_id','idx'])[['value']].sum()
    d_events.reset_index(inplace = True)
    #LAB count group
    l_events = idx_events[idx_events['event_type']=='L']
    l_events = l_events.groupby(['patient_id','idx'])[['value']].count()
    l_events.reset_index(inplace = True)
    #concatenate data frames
    aggregated_events = pd.concat([d_events,l_events]).reset_index(drop = True)   
    aggregated_events.columns = ['patient_id','feature_id','feature_value']
    aggregated_events['feature_value'] = aggregated_events['feature_value'].round(6)
    pivot = aggregated_events.pivot(index='patient_id', columns='feature_id', values='feature_value')
    norm = pivot/pivot.max()
    aggregated_events = pd.melt(norm.reset_index(), id_vars='patient_id',
                                value_name='feature_value').dropna()
    #export to csv 
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv',index = False)
    
    return aggregated_events  

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    Two dictionaries:
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    #create patient_features dictionary
    aggregated_events['zipped'] = list(zip(aggregated_events['feature_id'],aggregated_events['feature_value']))
    patient_features = {k : list(v) for k,v in aggregated_events[['patient_id','zipped']].groupby('patient_id')['zipped']}
    
    #create mortality dictionary
    all_ids = aggregated_events['patient_id'].unique()
    dead_ids = list(mortality['patient_id'])
    mortality = dict([(id, int(id in dead_ids)) for id in list(all_ids)])

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    1. op_file - which saves the features in svmlight format. 
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...    
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    for patient, features in patient_features.items():
        features = pd.DataFrame(features).sort_values(0)

        features = features.values.tolist()

        line1 = "{} {} \n".format(mortality.get(patient, 0), utils.bag_to_svmlight(features))
        
        line2 = "{} {} {} \n".format(int(patient),mortality.get(patient, 0), utils.bag_to_svmlight(features))
        
        deliverable1.write(bytes(line1,'UTF-8'))
        deliverable2.write(bytes(line2,'UTF-8'))

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()