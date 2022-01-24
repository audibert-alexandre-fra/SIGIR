import torch.utils.data as data
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocessing, data_preparation
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from model import HAHNN
    

def collate_batch(batch: list):
    """[Standard collate function to have text with different 
     padding]

    Args:
        batch ([liste of tensor]): [List of embedding tensor]

    Returns:
        [type]: [X_train, Y_train]
    """
    X_train = []
    Y_train = []
    for sample in batch:
        X_train.append(sample[0])
        Y_train.append(sample[1])
        
    X_train = pad_sequence(X_train, batch_first = True, padding_value = 0)
    Y_train = torch.stack(Y_train)
    return X_train, Y_train


def collate_batch_len(batch: list):
    """[Standard collate if you want to truncate]

    Args:
        batch (list): [List of embedding tensor]

    Returns:
        [type]: [X_train, Y_train]
    """
    X_train = []
    Y_train = []
    len_list = []
    for sample in batch:
        X_train.append(sample[0])
        len_list.append(len(sample[0]))
        Y_train.append(sample[1])
        
    X_train = pad_sequence(X_train, batch_first = True, padding_value = 0)
    Y_train = torch.stack(Y_train)
    return X_train, len_list, Y_train


class DataSet_1(Dataset):
    """[Create a standard dataset]

    Args:
        Dataset ([torch.utils.data.Dataset]): [Standard dataset for pytorch]
    """
    def __init__(self, x, y):
        super(DataSet_1, self).__init__()
        self.x = x
        self.y = y
        if len(self.x) != len(self.y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
    
def main():
    pass

if __name__ == "__main__":
    main()
                