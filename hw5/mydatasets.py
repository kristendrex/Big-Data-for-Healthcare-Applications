import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from scipy.sparse import csr_matrix
import pickle

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
    """
    :param path: a path to the seizure data CSV file
    :return dataset: a TensorDataset consists of a data Tensor and a target Tensor
    """
    # TODO: Read a csv file from path.
    # TODO: Please refer to the header of the file to locate X and y.
    # TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
    # TODO: Remove the header of CSV file of course.
    # TODO: Do Not change the order of rows.
    # TODO: You can use Pandas if you want to.
    df = pd.read_csv(path)
    y = (df['y']-1).values
    X = df.loc[:, 'X1':'X178'].values
    if model_type == 'MLP':
        dataset = TensorDataset(torch.from_numpy(X.astype('float32')), torch.from_numpy(y).type(torch.LongTensor))
    elif model_type == 'CNN':
        dataset = TensorDataset(torch.from_numpy(X.astype('float32')).unsqueeze(1), torch.from_numpy(y).type(torch.LongTensor))
    elif model_type == 'RNN':
        dataset = TensorDataset(torch.from_numpy(X.astype('float32')).unsqueeze(2), torch.from_numpy(y).type(torch.LongTensor))
    else:
        raise AssertionError("Wrong Model Type!")
    return dataset


def calculate_num_features(seqs):
    """
    :param seqs:
    :return: the calculated number of features
    """
    # TODO: Calculate the number of features (diagnoses codes in the train set)
    maxVal = 0
    for seq in seqs:
        for s in seq:
            if len(s) == 0:
                continue
            elif max(s) > maxVal:
                maxVal = max(s)
    maxVal = maxVal + 1
    print(maxVal)
    return maxVal
    #num_features = sequences[0].size()[1]
    #num_features = len(pickle.load(open("../data/mortality/processed/mortality.codemap.train", 'rb')))
    #return num_features


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels
        
        # TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
        # TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
        # TODO: You can use Sparse matrix type for memory efficiency if you want.
        def constructMatrix(aSeq, num_features):
            numVisits = len(aSeq)
            matrix = []
            for i in range(numVisits):
                row = [0] * num_features
                for diag in aSeq[i]:
                    row[int(diag)] = 1
                matrix.append(row)
            return sparse.csr_matrix(np.array(matrix))
            
        self.seqs = [constructMatrix(seq,num_features) for seq in seqs]
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
    seqs, lenList, labelList = [], [], []
    tmp = [(index, data.shape[0]) for index, (data, _) in enumerate(batch)]
    sorted_tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    max_row = sorted_tmp[0][1] #2
    max_col = batch[0][0].shape[1] #911
    for index, length in sorted_tmp:
        lenList.append(length)
        labelList.append(batch[index][1])
        data = sparse.csr_matrix(batch[index][0]).toarray()
        add_num_rows = max_row - length
        if add_num_rows > 0:
            added_rows = np.zeros((add_num_rows, max_col))
            final_matrix = np.concatenate((data, added_rows))
            seqs.append(final_matrix)
        else:
            seqs.append(data)
    seqs_tensor = torch.FloatTensor(seqs)
    lengths_tensor = torch.LongTensor(lenList)
    labels_tensor = torch.LongTensor(labelList)
    return (seqs_tensor, lengths_tensor), labels_tensor
            
                      
                      
                      
                      
                      
        
