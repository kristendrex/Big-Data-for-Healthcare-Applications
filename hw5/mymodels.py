import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np


checksum = '169a9820bbc999009327026c9d76bcf1'

class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        #self.hidden = nn.Linear(178, 16)
        #self.out = nn.Linear(16,5)
        ## improvements:
        self.l1 = nn.Linear(178, 64)
        self.l2 = nn.Linear(64,5)

    def forward(self, x):
        #x = torch.sigmoid(self.hidden(x))
        #out = self.out(x)
        x = F.relu(self.l1(x))
        out = self.l2(x)
        return out


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
        self.cl1 = nn.Linear(in_features=16 * 41, out_features=128)
        self.cl2 = nn.Linear(128, 64)
        self.cl3 = nn.Linear(64, 5)

        #improvement
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv1(x)))
        #improvement
        x = self.pool(F.relu(self.dropout(self.conv1(x))))
        x = self.pool(F.relu(self.dropout(self.conv2(x))))
        x = x.view(-1, 16 * 41)
        x = F.relu(self.dropout(self.cl1(x)))
        x = F.relu(self.dropout(self.cl2(x)))
        #x = F.relu(self.cl1(x))
        #x = F.relu(self.cl2(x))
        x = self.cl3(x)
        return x

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=1, dropout=0.5, batch_first=True)
        self.r1 = nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        #return x
        x, _ = self.rnn(x)
        x = self.r1(x[:, -1, :])
        return x


class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        self.fc1 = nn.Linear(in_features=dim_input, out_features=32)
        self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, input_tuple):
        aSeqs, lengths = input_tuple
        aSeqs = torch.tanh(self.fc1(aSeqs))
        packed = pack_padded_sequence(aSeqs, lengths, batch_first=True)
        packed, idk = self.rnn(packed)
        packed, _ = pad_packed_sequence(packed, batch_first=True)
        out = self.fc2(packed[:, -1, :])
        return out
