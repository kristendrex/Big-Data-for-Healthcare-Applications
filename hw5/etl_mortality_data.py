import numpy as np
import os
import pickle
import pandas as pd

checksum = '169a9820bbc999009327026c9d76bcf1'

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
    """
    :param icd9_object: ICD-9 code (Pandas/Numpy object).
    :return: extracted main digits of ICD-9 code
    """
    icd9_str = str(icd9_object)
    # Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
    converted = icd9_str
    if icd9_str[0] == 'E':
        converted = icd9_str[0:4]
    else:
        converted = icd9_str[0:3]
    return converted

def build_codemap(df_icd9, transform):
    """
    :return: Dict of code map {main-digits of ICD9: unique feature ID}
    """
    # Build a code map using ONLY train data
    df_icd9['ICD9_CODE'].dropna(inplace=True)
    df = df_icd9['ICD9_CODE'].apply(transform)
    codes = df.unique()
    codes = [x for x in codes if str(x) != 'nan']
    codemap = dict(zip(codes, np.arange(len(codes))))
    return codemap


def create_dataset(path, codemap, transform):
    """
    :param path: path to the directory contains raw files.
    :param codemap: 3-digit ICD-9 code feature map
    :param transform: e.g. convert_icd9
    :return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
    """
    seq_data = []
    patient_ids = []
    labels = []
    # Load data from the three csv files
    df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
    df_admission = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
    df_diagnoses = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
    
    # Convert diagnosis code in to unique feature ID.
    df_diagnoses['ICD9_CODE'] = df_diagnoses['ICD9_CODE'].transform(transform)
    
    # Group the diagnosis codes for the same visit.
    diag_visit = df_diagnoses.groupby(['HADM_ID']) 
    
    # Group the visits for the same patient.
    pat_visit = df_admission.groupby(['SUBJECT_ID'])
    
    # Make a visit sequence dataset as a List of patient Lists of visit Lists
    for patient, visits in pat_visit:
        patient_ids.append(patient)
        dates = visits.sort_values(by=['ADMITTIME'])
        patient_diagnoses = []
        for i, each in dates.iterrows():
            diagnoses = diag_visit.get_group(each['HADM_ID'])['ICD9_CODE'].values
            patient_diagnoses.append([codemap[x] for x in diagnoses if x in codemap])
        seq_data.append(patient_diagnoses)
        
    # Make patient-id List and label List also.
    for i,p in pat_visit:
        labels.append(df_mortality.loc[df_mortality['SUBJECT_ID'] == i]['MORTALITY'].values[0])
    return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
