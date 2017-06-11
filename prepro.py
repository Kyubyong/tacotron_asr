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
import csv
import codecs
from data import text2idx

def make_train_data():
    import csv
    
    sound_fpaths, converteds, texts = [], [], [] 
    reader = csv.reader(codecs.open(hp.web + "/text.csv", 'rb', 'utf-8'))
    for row in reader:
        sound_fname, text, duration = row
        sound_fpath = hp.web + "/" + sound_fname + ".wav"
        cleaned, converted = text2idx(text)
        if (len(text) <= hp.max_len):
            sound_fpaths.append(sound_fpath)
            converteds.append(np.array(converted, np.int32).tostring())
            texts.append(cleaned)

    # Split into train and eval. We reserve the last mini-batch for evaluation
    X_train, Y_train = sound_fpaths[:-hp.batch_size], converteds[:-hp.batch_size]
    X_eval, Y_eval = sound_fpaths[-hp.batch_size:], texts[-hp.batch_size:]
    
    # Save
    pickle.dump((X_train, Y_train), open('data/train.pkl', 'wb'))
    pickle.dump((X_eval, Y_eval), open('data/eval.pkl', 'wb'))
    
if __name__ == "__main__":
    make_train_data()
    print("Done!")            
       