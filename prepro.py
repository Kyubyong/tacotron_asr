# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron_asr
'''
from __future__ import print_function

from collections import Counter
import glob
import os
import pickle
import re

import librosa
from tqdm import tqdm 

from hyperparams import Hyperparams as hp
import numpy as np


def make_info():
    if not os.path.exists('preprocessed'): os.mkdir('data')
    with open('preprocessed/info.txt', 'w') as fout:
        for text_fpath in tqdm(glob.glob(hp.vctk)):
            # text
            text = open(text_fpath, 'r').read().strip()
            
            # sound
            sound_fpath = text_fpath.replace("/txt", "/wav48").replace("txt", "wav")
            y, sr = librosa.load(sound_fpath)
            duration = librosa.get_duration(y, sr)
            
            fout.write("%s\t%.2f\t%s\n" % (sound_fpath, duration, text))

def make_data():
    from data import load_vocab, text2idx
    
    # Load vocabulary
    token2idx, idx2token = load_vocab() 
    
    # Make sound file paths and their texts  
    sound_fpaths, texts_converted, texts = [], [], []
    for line in tqdm(open('preprocessed/info.txt', 'r')):
        try:
            sound_fpath, duration, text = line.strip().split('\t')
        except ValueError:
            continue
        
        cleaned, converted = text2idx(text)
        if (len(converted) <= hp.max_len) and (1. < float(duration) <= hp.max_duration):
            sound_fpaths.append(sound_fpath)
            texts_converted.append(np.array(converted, np.int32).tostring())
            texts.append(cleaned)
    
    # Split into train and eval
    X_train, Y_train = sound_fpaths[:-10*hp.batch_size], texts_converted[:-10*hp.batch_size]
    X_eval, Y_eval = sound_fpaths[-10*hp.batch_size:], texts[-10*hp.batch_size:]
    
    # Save
    pickle.dump((X_train, Y_train), open('preprocessed/train.pkl', 'wb'))
    pickle.dump((X_eval, Y_eval), open('preprocessed/eval.pkl', 'wb'))

if __name__ == "__main__":
#     print("Making info file... Be patient! This might take more than 30 minutes!")
#     make_info()

    print("Making training/evaluation data...")
    make_data()
    print("Done!")            
       