# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron_asr
'''

from __future__ import print_function
import os
import re
import numpy as np
import pickle
from hyperparams import Hyperparams as hp


def load_vocab():
    vocab = "ES abcdefghijklmnopqrstuvwxyz'" # E: Empty
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    return char2idx, idx2char


   
                
def text2idx(text):
    # Load vocabulary
    char2idx, idx2char = load_vocab() 
    
    # Convert
    text = re.sub(r"[^ a-z']", "", text.lower()).strip() + "S"
    converted = [char2idx[char] for char in text]
    return text, converted
        
def load_train_data():
    """We train on the whole data but the last mini-batch."""
    
    sound_fpaths, converteds = pickle.load(open('data/train.pkl', 'rb'))
    
    return sound_fpaths, converteds
 
def load_eval_data():
    from utils import get_spectrogram, reduce_frames
    """We evaluate on the last mini-batch."""
    sound_fpaths, texts = pickle.load(open('data/eval.pkl', 'rb'))
    # Extract spectrogram from sound_fpaths
    char2idx, idx2char = load_vocab() 
    
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
        
    return X, texts # 3d array, list of str 
 

