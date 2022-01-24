# !/usr/bin/python3 
import gensim
import torch
import os
from d2l import torch as d2l
import numpy as np


def download_pretrained_vector(current_absolute_path: str):
    """[Return a pretrained law2vec thanks to an absolute current path]

    Args:
        current_absolute_path (str): [current absolute path
        often os.path.dirname(os.path.abspath(__file__))]

    Returns:
        [gensim.model]: [Pretrained law2vec]
    """
    path = "./law2vec_weight/Law2Vec.200d.txt"
    absolute_path_to_model = os.path.join(os.path.dirname(os.path.abspath(os.path.abspath(__file__))), path)
    correct_path = os.path.relpath(absolute_path_to_model, current_absolute_path)
    pretrained_vector = gensim.models.KeyedVectors.load_word2vec_format(correct_path, binary=False)
    return pretrained_vector


def create_pretrained_emb_matrix(vocab: dict, current_absolute_path: str):
    """[Create pretrained embedded matrix for Deep learning model,
    which matchs with the vocab]

    Args:
        vocab (dict): [dictionnay of known tokens]
        current_absolute_path (str): [current absolute path
        often os.path.dirname(os.path.abspath(__file__))]

    Returns:
        [np.ndarray]: [description]
    """
    law_to_vec = download_pretrained_vector(current_absolute_path)
    nb_token = len(vocab.idx_to_token)
    print(" The vocabulary is composed of {0}".format(nb_token))
    for indice, token in enumerate(vocab.idx_to_token):
        if token not in law_to_vec:
            if token == '<pad>':
                law_to_vec.add_vector(token, np.zeros(200))
            else:
                law_to_vec.add_vector(token, np.random.normal(scale=0.6, size=(200, )))
    pretrained_matrix = law_to_vec[vocab.idx_to_token]
    return pretrained_matrix
    
    
def main():
    pass


if __name__ == '__main__':
    main()
    