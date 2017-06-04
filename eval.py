# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function
import codecs
import copy
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from train import Graph
from data import load_vocab, load_eval_data

def eval(): 
    # Load graph
    g = Graph(is_training=False); print("Graph loaded")
    
    # Load data
    X, Y = load_eval_data() # texts
    token2idx, idx2token = load_vocab()
            
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            
            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
              
            timesteps = hp.max_len # Adjust this number as you want
            outputs_shifted = np.zeros((hp.batch_size, timesteps), np.int32)
            outputs = np.zeros((hp.batch_size, timesteps), np.int32)   # hp.n_mels*hp.r  
            for j in range(timesteps):
                # predict next frames
                _outputs = sess.run(g.outputs, {g.x: X, g.y: outputs_shifted})
                _preds = np.argmax(_outputs, axis=-1)
                # update character sequence
                if j < timesteps - 1:
                    outputs_shifted[:, j + 1] = _preds[:, j]
                outputs[:, j] = _preds[:, j]
             
    
    # Generate wav files
    if not os.path.exists('samples'): os.mkdir('samples') 
    with codecs.open('samples/text.txt', 'w', 'utf-8') as fout:
        for i, (expected, got) in enumerate(zip(Y, outputs)):
            # write text
            fout.write("Expected: {}\n".format(expected))
#             fout.write("Got     : {}\n\n".format(("".join(idx2token[idx] for idx in np.fromstring(got, np.int32)).split("S")[0])))
            fout.write("Got     : {}\n\n".format(("".join(idx2token[idx] for idx in np.fromstring(got, np.int32)))))
            
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
