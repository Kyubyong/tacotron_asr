# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function
import os
import re
import numpy as np
import pickle
from hyperparams import Hyperparams as hp


def load_vocab():
    assert os.path.exists('preprocessed/vocab.txt'), "Did you make vocabulary file?" 
    
    vocab = open('preprocessed/vocab.txt', 'r').read().splitlines()
    
    token2idx = {token:idx for idx, token in enumerate(vocab)}
    idx2token = {idx:token for idx, token in enumerate(vocab)}
    
    return token2idx, idx2token

def text2idx(text):
    # Load vocabulary
    token2idx, idx2token = load_vocab() 
    
    # Convert
    converted = []
    text = re.sub(r"[^ a-z']", "", text.lower()).strip() + " S"
    for word in text.split():
        if word in token2idx:
            converted.append(token2idx[word])
            converted.append(2) # 2: space
        else:
            converted.extend([token2idx[char] for char in word])
            converted.append(2) # 2: space
    return text, converted[:-1] # -1: extra space
        
def load_train_data():
    """We train on the whole data but the last mini-batch."""
    X_train, Y_train = pickle.load(open('preprocessed/train.pkl', 'rb'))
    return X_train, Y_train
 
def load_eval_data():
    from utils import get_spectrogram, reduce_frames
    """We evaluate on the last mini-batch."""
    sound_fpaths, gt = pickle.load(open('preprocessed/eval.pkl', 'rb')) # gt: ground truth
    
    # Extract spectrogram from sound_fpaths
    token2idx, idx2token = load_vocab() 
    
    xs, maxlen = [], 0
    for sound_fpath in sound_fpaths:
        spectrogram = get_spectrogram(sound_fpath)
        x = reduce_frames(spectrogram, hp.r)
        maxlen = max(maxlen, len(x))
        xs.append(x)
        
    # Set the length of samples in X to the maximum among them.
    X = np.zeros(shape=(len(xs), maxlen, hp.n_mels*hp.r), dtype=np.float32)
    for i, x in enumerate(xs):
        X[i, :len(x), :] = x
        
    return X, gt # 3d array, list of str 
 

