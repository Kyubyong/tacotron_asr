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
    X, Y = load_eval_data()
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
                for num in range(len(X)//hp.batch_size):
                    print("{}/{} is working".format(num+1, len(X)//hp.batch_size))
                          
                    x = X[num*hp.batch_size:(num+1)*hp.batch_size]
                    gt = Y[num*hp.batch_size:(num+1)*hp.batch_size]
                    
                    outputs = np.zeros((hp.batch_size, hp.max_len), np.int32)  
                    for j in range(hp.max_len):
                        # predict next character
                        _outputs = sess.run(g.outputs, {g.x: x, g.y: outputs})
    #                     if i==0 and j==0: 
    #                         aa, bb, cc, dd, ee, ff, gg = copy.deepcopy(prenet_out),  copy.deepcopy(decoder_inputs), copy.deepcopy(memory), copy.deepcopy(_outputs), X, copy.deepcopy(outputs), copy.deepcopy(dec) 
    #                     if i==1 and j==0: 
    #                         print(np.array_equiv(prenet_out, aa))
    #                         print(np.array_equiv(decoder_inputs, bb))
    #                         print(np.array_equiv(memory, cc))
    #                         
    #                         print(np.array_equiv(_outputs, dd))
    #                         print(np.array_equiv(X, ee))
    #                         print(np.array_equiv(outputs, ff))
    #                         print(np.array_equiv(dec, gg))
                        _preds = np.argmax(_outputs, axis=-1)
                        
                        # update character sequence
                        outputs[:, j] = _preds[:, j]
                
                    # Write to file
                    for i, (expected, got) in enumerate(zip(gt, outputs)): # ground truth vs. prediction
                        fout.write("Expected: {}\n".format(expected.split("S")[0]))
#                         fout.write("Expected     : {}\n\n".format(("".join(idx2char[idx] for idx in np.fromstring(expected, np.int32))).split("S")[0]))
                        fout.write("Got     : {}\n\n".format(("".join(idx2char[idx] for idx in np.fromstring(got, np.int32))).split("S")[0]))
                        fout.flush()
#                         converted = []
#                         for _output in outputs:
#                             converted.append("".join(idx2char[idx] for idx in _output))
#                         print("\n".join(converted))
#                         
#                         converted = []
#                         for _output in _preds:
#                             converted.append("".join(idx2char[idx] for idx in _output))
#                         print("================")
#                         print("\n".join(converted))
        
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
