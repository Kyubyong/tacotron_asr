# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron_asr
'''

from __future__ import print_function

import codecs
import os

from data import load_vocab, load_eval_data, load_train_data
from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph


def eval(): 
    # Load graph
    g = Graph(is_training=False); print("Graph loaded")
    
    # Load data
    x, y = load_eval_data()
    char2idx, idx2char = load_vocab()
            
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]
            
            # Speech to Text
            if not os.path.exists('samples'): os.mkdir('samples') 
            with codecs.open('samples/{}.txt'.format(mname), 'w', 'utf-8') as fout:
                preds = np.zeros((hp.batch_size, hp.max_len), np.int32)
                for j in range(hp.max_len):
                    _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                    preds[:, j] = _preds[:, j]
                
                # Write to file
                for i, (expected, got) in enumerate(zip(y, preds)): # ground truth vs. prediction
                    fout.write("Expected: {}\n".format(expected.split("S")[0]))
                    fout.write("Got     : {}\n\n".format(("".join(idx2char[idx] for idx in np.fromstring(got, np.int32))).split("S")[0]))
                    fout.flush()
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
