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
            with codecs.open('samples/{}_beam_width_{}.txt'.format(mname, hp.beam_width), 'w', 'utf-8') as fout:
                for num in range(len(X)//hp.batch_size):
                    print("{}/{} is working".format(num+1, len(X)//hp.batch_size))
                    fout.write("batch_{}\n".format(num))      
                    x = X[num*hp.batch_size:(num+1)*hp.batch_size]
                    gt = Y[num*hp.batch_size:(num+1)*hp.batch_size]
                    
                    if hp.beam_width==1:
                        preds = np.zeros((hp.batch_size, hp.max_len), np.int32)
                        for j in range(hp.max_len):
                            _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                            preds[:, j] = _preds[:, j]
                    else: # beam decode    
                        ## first step  
                        preds = np.zeros((hp.beam_width*hp.batch_size, hp.max_len), np.int32)  #  (bw*N, T)
                        logprobs = sess.run(g.logprobs, {g.x: x, g.y: np.zeros((hp.batch_size, hp.max_len), np.int32)}) # (N, T, V)
                        target = logprobs[:, 0, :] # (N, V)
                        
                        preds_in_beam = target.argsort()[:, ::-1][:, :hp.beam_width].flatten() # (bw*N,)
                        preds[:, 0] = preds_in_beam
                        
                        logp_in_beam = np.sort(target)[:, ::-1][:, :hp.beam_width].flatten() # (bw*N,)
                        logp_in_beam = np.repeat(logp_in_beam, hp.beam_width, axis=0) # (bw*bw*N, )
                         
                        ## remaining steps
                        for i in range(1, hp.max_len-1):
                            logprobs = sess.run(g.logprobs, {g.x: np.repeat(x, hp.beam_width, 0), g.y: preds}) # (bw*N, T, V)
                            target = logprobs[:, i, :] # (bw*N, V)
                             
                            preds_in_beam = target.argsort()[:, ::-1][:, :hp.beam_width].flatten() # (bw*bw*N,)
                            logp_in_beam += np.sort(target)[:, ::-1][:, :hp.beam_width].flatten() # (bw*bw*N, )
     
                            preds = np.repeat(preds, hp.beam_width, axis=0) # (bw*bw*N, T) <- Temporary shape expansion
                            preds[:, i] = preds_in_beam
                                   
                            elems = [] # (bw*N). bw elements are selected out of bw^2
                            for j, cluster in enumerate(np.split(logp_in_beam, hp.batch_size)): # cluster: (bw*bw,)
                                if i == hp.max_len-2: # final step
                                    elem = np.argsort(cluster)[::-1][:1] # final 1 best
                                    elems.extend(list(elem + j*len(cluster)))
                                else:
                                    elem = np.argsort(cluster)[::-1][:hp.beam_width]
                                    elems.extend(list(elem + j*len(cluster)))
                            preds = preds[elems] # (N, T) if final step,  (bw*N, T) otherwise. <- shape restored
                            logp_in_beam = logp_in_beam[elems]
                            logp_in_beam = np.repeat(logp_in_beam, hp.beam_width, axis=0) # (bw*bw*N, )
                            
                            for l, pred in enumerate(preds[:hp.beam_width]):
                                fout.write(str(l) + " " + u"".join(idx2char[idx] for idx in pred).split("S")[0] + "\n")
                
                    # Write to file
                    for i, (expected, got) in enumerate(zip(gt, preds)): # ground truth vs. prediction
                        fout.write("Expected: {}\n".format(expected.split("S")[0]))
#                         fout.write("Expected     : {}\n\n".format(("".join(idx2char[idx] for idx in np.fromstring(expected, np.int32))).split("S")[0]))
                        fout.write("Got     : {}\n\n".format(("".join(idx2char[idx] for idx in np.fromstring(got, np.int32))).split("S")[0]))
                        fout.flush()
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
