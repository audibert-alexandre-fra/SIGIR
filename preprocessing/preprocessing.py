# !/usr/bin/python3 
from typing import List
from d2l import torch as d2l
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
from math import floor
import re
from nltk.corpus import wordnet
import nltk

to_remove = ['he', 'she', 'who', 'be', 'have', 'again', 'more', 'no', 'not', 'only', 'too', 'can', 'just']
new_stopwords = set(stopwords.words('english')).difference(to_remove)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def isalpha_or_dot(words: str):
    """[Return True if all characters in the string are alphabets
        or if the charactere is an end of sentence]

    Args:
        words (str): [String to check]

    Returns:
        [bool]: [True if all characters in the string are alphabets
        or is an end of sentence]
    """
    return (words.isalpha() and len(words) > 1) or words == '.' \
        or words == '?' or words == '!'


def clean_txt(text: str):
    """[Clean a string str of all invalide token: number and 
    punctuation whitout dots]

    Args:
        text (str): [String to checlk]

    Returns:
        [str]: [String which has no numbers and no punctuation, 
        but only keeps letters and dots]
    """
    text = re.sub('[^A-Za-z\.\?\!]+', ' ', text)
    return text


def preprocessing(data: np.ndarray, lower_case: bool = None, 
                  stop_words: bool = None, lemmatisation: bool = None):
    """[Preprocesssing function: Tokenization of the documents and
    it's possible to add lower_case, remove stop words and lemmatisation]

    Args:
        data (np.ndarray): [array which contains the facts of our case]
        lower_case (bool, optional): [True for of only lower_case]. Defaults to None/False.
        stop_words (bool, optional): [True to remove stopwords]. Defaults to None/False.
        lemmatisation (bool, optional): [True to use a lemmatizer]. Defaults to None/False.

    Returns:
        [type]: [description]
    """
    if lower_case is None:
        lower_case = False
        
    if stop_words is None:
        stop_words = False
        
    if lemmatisation is None:
        lemmatisation = False
    
    if lemmatisation == True:
        lemmatiser = WordNetLemmatizer()
        
    vfunc = np.vectorize(clean_txt)    
    data = vfunc(data)
    
    train_tokens = d2l.tokenize(data, token = 'word')
    dataset_tokenise = []
    
    for document in train_tokens:
        if lower_case and stop_words and lemmatisation:
            tokens = [lemmatiser.lemmatize(word.lower(), get_wordnet_pos(word.lower())) for word in document 
                     if isalpha_or_dot(word) and lemmatiser.lemmatize(word.lower(), get_wordnet_pos(word.lower())) not in new_stopwords]
        elif lower_case and stop_words:
            tokens = [word.lower() for word in document 
                     if isalpha_or_dot(word) and word not in new_stopwords]
        elif lemmatisation and stop_words:
            tokens = [lemmatiser.lemmatize(word.lower(), get_wordnet_pos(word.lower())) for word in document 
                     if isalpha_or_dot(word) and lemmatiser.lemmatize(word.lower(), get_wordnet_pos(word.lower())) not in new_stopwords]
        elif lower_case and lemmatisation:
            tokens = [lemmatiser.lemmatize(word) for word in document 
                     if isalpha_or_dot(word)]
        elif lower_case:
            tokens = [word.lower() for word in document 
                     if isalpha_or_dot(word)]
        elif lemmatisation:
            tokens = [lemmatiser.lemmatize(word) for word in document 
                     if isalpha_or_dot(word)]
        elif stop_words:
            tokens = [word for word in document 
                     if isalpha_or_dot(word) and word not in new_stopwords]
        else:
            tokens = [word for word in document 
                     if isalpha_or_dot(word)]
        dataset_tokenise.append(tokens)
    return dataset_tokenise


def find_great_size_pad(current_length: int, max_len: int):
    """[Return the number of neccessary padding to have 
        current_length + padding mod max_len equal to 0]

    Args:
        current_length ([int]): [original length]
        max_len ([int]): [contraint of padding]

    Returns:
        [int]: [number of necessary padding]
    """
    number_padding = (current_length // max_len + 1) * (max_len) - current_length
    return number_padding


def doc_to_sentence(data: list, max_len_sentence: int):
    """[Each sentences are padding to attain max_en_sentence
    if a sentence is longer than max_en_sentence it's cut into
    chunk of size max_en_sentence]

    Args:
        data ([list]): [List of string which have to be preprocess]
        max_len_sentence ([int]): [maximum number of words autorize in a setence]

    Returns:
        [list]: [New dataset]
    """
    dataset = []
    for text in data:
        doc = []
        index_start_sentence = 0
        length_current_sentence = 0
        index = 0
        while(index < len(text)):
            if text[index] == '.' or text[index] == '!' or text[index] == '?' or index == (len(text)-1):
                length_current_sentence = index - index_start_sentence + 1
                if length_current_sentence <= max_len_sentence:
                    doc.extend(text[index_start_sentence: index + 1] + 
                               ["<pad>"]*(max_len_sentence - length_current_sentence))
                    index_start_sentence = index + 1
                else:
                    doc.extend(text[index_start_sentence: index+1] +
                               ["<pad>"]*find_great_size_pad(length_current_sentence, max_len_sentence))

                    index_start_sentence = index + 1
            index += 1
        dataset.append(doc)
    return dataset
            

def create_vocabulary(dataset_tokenise: list, min_freq: int = None):
    """[Create a dictionnary which represents the vocabulary]

    Args:
        dataset_tokenise (list): [Preprocessing dataset]
        min_freq (int, optional): [The number of words which more than 5 times]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if min_freq is None:
        min_freq = 5
    vocab = d2l.Vocab(dataset_tokenise, min_freq = min_freq, reserved_tokens = ['<pad>'])
    return vocab


def read_dataset(current_absolute_path: str):
    """[Return dataset]
    Args:
        current_absolute_path (str): [Absolut Folder Path]
    Returns:
        [Dataframe Multi_index]: [Return the entire dataset]
    """
    path = "./data/data.pkl"
    chaine_2 = os.path.join(os.path.dirname(os.path.abspath(os.path.abspath(__file__))), path)
    data = pd.read_pickle(os.path.relpath(chaine_2, current_absolute_path))
    return data


def main():
    pass

    
if __name__ == '__main__':
    main()