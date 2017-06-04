# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
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

def make_vocab():
    # make character vocab first
    vocab = "ES abcdefghijklmnopqrstuvwxyz'" # E: Empty, S: EOS
    vocab = [elem for elem in vocab]
    
    # append common words to vocab
    assert os.path.exists('preprocessed/info.txt'), "Did you run `prepro.py`?"
    
    word2cnt = dict()
    for line in tqdm(open('preprocessed/info.txt', 'r')):
        try:
            _, _, sent = line.split("\t")
            text = re.sub(r"[^ a-z']", "", sent.lower()).strip()
            for word in text.split():
                if word in word2cnt:
                    word2cnt[word] += 1
                else:
                    word2cnt[word] = 1
        except ValueError:
            print(line)
            continue
            
    c = Counter(word2cnt)
    t = c.most_common(hp.max_word_vocab_size)
    vocab = vocab + [word for word, cnt in c.most_common(hp.max_word_vocab_size) if word not in vocab] 
    
    # write vocab to a file
    with open("preprocessed/vocab.txt", "w") as fout:
        fout.write("\n".join(vocab))

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
            print("Oops! Seems that this line has some problem => ", line)
            continue
        
        cleaned, converted = text2idx(text)
        if (len(converted) <= hp.max_len) and (1. < float(duration) <= hp.max_duration):
            sound_fpaths.append(sound_fpath)
            texts_converted.append(np.array(converted, np.int32).tostring())
            texts.append(cleaned)
    
    # Split into train and eval
    X_train, Y_train = sound_fpaths[:-hp.batch_size], texts_converted[:-hp.batch_size]
    X_eval, Y_eval = sound_fpaths[-hp.batch_size:], texts[-hp.batch_size:]
    
    # Save
    pickle.dump((X_train, Y_train), open('preprocessed/train.pkl', 'wb'))
    pickle.dump((X_eval, Y_eval), open('preprocessed/eval.pkl', 'wb'))

if __name__ == "__main__":
#     print("Making info file... Be patient! This might take more than 30 minutes!")
#     make_info()
#     
#     print("Making vocabulary...")
#     make_vocab()

    print("Making training/evaluation data...")
    make_data()
    print("Done!")            
       