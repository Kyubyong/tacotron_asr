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

def make_data():
    sound_fpaths, converteds, texts, multi_texts = [], [], [], set()
    for text_fpath in tqdm(glob.glob(hp.vctk)):
        
        # text
        text = open(text_fpath, 'r').read().strip()
        text, converted = text2idx(text)
        
        if len(text) <= hp.max_len:
            # sound
            sound_fpath = text_fpath.replace("/txt", "/wav48").replace("txt", "wav")
            
            if text in texts:
                
                multi_texts.add(text)
            sound_fpaths.append(sound_fpath)
            converteds.append(np.array(converted, np.int32).tostring())
            texts.append(text)
    #print(len(texts))
    s_train, c_train = [], []
    s_eval, t_eval = [], []
    for s, c, t in zip(sound_fpaths, converteds, texts):
        if t not in multi_texts and len(s_eval) < 10*hp.batch_size:
            s_eval.append(s)
            t_eval.append(t)
            
        else:
            s_train.append(s)
            c_train.append(c)
    #print(c_train[0])
    #print("\n".join(t_eval[:10]))        
    pickle.dump((s_train, c_train), open('data/train.pkl', 'wb'))
    pickle.dump((s_eval, t_eval), open('data/eval.pkl', 'wb'))

if __name__ == "__main__":
#     print("Making info file... Be patient! This might take more than 30 minutes!")
#     make_info()

    print("Making training/evaluation data...")
    make_data()
    print("Done!")            
       