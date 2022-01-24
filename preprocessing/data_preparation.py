# !/usr/bin/python3 
from pandas.core.indexing import maybe_convert_ix
from . import preprocessing
from d2l import torch as d2l
import torch 
import os
from . import generate_pretrained


def text_format(vocab: dict, dataset: list):
    """[Create a dataset with no padding and each text will be representend
    like a block]

    Args:
        vocab (dict): [The vocabulary obtained during preprocessing]
        dataset ([type]): [list of tensor]

    Returns:
        [list of tensor]: [The final form of the dataset]]
    """
    train_features = [torch.tensor(vocab[line]) for line in dataset]
    return train_features


def sentence_format(vocab: dict, max_len_sentence: int, dataset: list):
    """[Create a dataset with padding and each text will be representend
    like multi-chunks]

    Args:
        vocab (dict): [The vocabulary obtained during preprocessing]
        max_len_sentence ([int]): [length of futur chunk]
        dataset ([list]): [original dataset]

    Returns:
        [list of tensor]: [The final form of the dataset]
    """
    dataset = preprocessing.doc_to_sentence(dataset, max_len_sentence)
    train_features = [torch.tensor(vocab[line]) for line in dataset]
    return train_features


def data_preparation(path : str, min_freq: int=10,
                     max_len_sentence: int=None, remove_stop_words: bool=True):
    """[Last step of data preparation to obtain a valid dataset]

    Args:
        path (str): [current absolute path
        often os.path.dirname(os.path.abspath(__file__))]
        min_freq (int, optional): [The minimum number of time that a token has to be seen
        to be in the vocabulary]. Defaults to 10.
        max_len_sentence (int, optional): [The maximum number of token in a chunk, if none
        it will be a block layout]. Defaults to None.
        remove_stop_words (bool, optional): [True to remove unecessary token]. Defaults to True.

    Returns:
        [Union(list, torch.tensor): [list: represent train, test, validation, and
        torch.tensor represents the pretrained matrix]
    """

    data = preprocessing.read_dataset(path)
    X_train = data.loc['Train']["Fact"].values
    X_val = data.loc['Val']["Fact"].values
    X_test = data.loc['Test']["Fact"].values
    if max_len_sentence is None:
        print('====== Preparation Train Set ======')
        X_train = preprocessing.preprocessing(X_train, lower_case=True,
                                              stop_words=remove_stop_words, lemmatisation=True)
        vocab = preprocessing.create_vocabulary(X_train, min_freq=min_freq)
        X_train = text_format(vocab=vocab, dataset=X_train)
        X_train = [X_train,
                   torch.from_numpy(data.loc['Train'][['Violation', 'Quantum']].values)]
        print('====== Prepatation Validation Set ======')
        X_val = preprocessing.preprocessing(X_val, lower_case=True,
                                            stop_words=remove_stop_words, lemmatisation=True)
        X_val = text_format(vocab=vocab, dataset=X_val)
        X_val =  [X_val,
                  torch.from_numpy(data.loc['Val'][['Violation', 'Quantum']].values)]
        print('====== Preparation Test Set ======' )
        X_test =  preprocessing.preprocessing(X_test, lower_case=True,
                                              stop_words=remove_stop_words, lemmatisation=True)
        X_test = text_format(vocab=vocab, dataset=X_test)
        X_test = [X_test,
                  torch.from_numpy(data.loc['Test'][['Violation', 'Quantum']].values)]
        
        print("====== Pretrained Embedding Preparation ======")
        pretrained_matrix = generate_pretrained.create_pretrained_emb_matrix(vocab=vocab,
                                                                          current_absolute_path=path)

        print("====== Preprocessing Done ! ======")
        
    else:
        print('====== Preparation Train Set ======')
        X_train = preprocessing.preprocessing(X_train, lower_case=True,
                                              stop_words=remove_stop_words, lemmatisation=True)
        vocab = preprocessing.create_vocabulary(X_train, min_freq=min_freq)
        X_train = sentence_format(vocab=vocab, max_len_sentence=max_len_sentence, dataset=X_train)
        X_train = [X_train,
                   torch.from_numpy(data.loc['Train'][['Violation', 'Quantum']].values)]
        print('====== Prepatation Validation Set ======')
        X_val = preprocessing.preprocessing(X_val, lower_case=True,
                                            stop_words=remove_stop_words, lemmatisation=True)
        X_val = sentence_format(vocab=vocab, max_len_sentence=max_len_sentence, dataset=X_val)
        X_val =  [X_val,
                  torch.from_numpy(data.loc['Val'][['Violation', 'Quantum']].values)]
        print('====== Preparation Test Set ======' )
        X_test =  preprocessing.preprocessing(X_test, lower_case=True,
                                              stop_words=remove_stop_words, lemmatisation=True)
        X_test = sentence_format(vocab=vocab, max_len_sentence=max_len_sentence, dataset=X_test)
        X_test = [X_test,
                  torch.from_numpy(data.loc['Test'][['Violation', 'Quantum']].values)]
        
        print("====== Pretrained Embedding Preparation ======")
        pretrained_matrix = generate_pretrained.create_pretrained_emb_matrix(vocab=vocab,
                                                                          current_absolute_path=path)

        print("====== Preprocessing Done ! ======")   
    return ([X_train, X_val, X_test], torch.from_numpy(pretrained_matrix))


def data_preparation_bert(path : str):
    """[Return basic list of tensor str for bert. The rest of the preprocessing
    will be done in the dataloader_bert]

    Args:
        path (str): [current absolute path
        often os.path.dirname(os.path.abspath(__file__))]

    Returns:
        [type]: [list of tensor]
    """
    data = preprocessing.read_dataset(path)
    X_train = data.loc['Train']["Fact"].values
    X_val = data.loc['Val']["Fact"].values
    X_test = data.loc['Test']["Fact"].values
    
    Y_train = torch.from_numpy(data.loc['Train'][['Violation', 'Quantum']].values)
    Y_val = torch.from_numpy(data.loc['Val'][['Violation', 'Quantum']].values)
    Y_test = torch.from_numpy(data.loc['Test'][['Violation', 'Quantum']].values)
    return [[X_train, Y_train], [X_val, Y_val], [X_test, Y_test]]                         


def main():
    pass
    
    
if __name__ == "__main__":
    main()