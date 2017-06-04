# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import os

import librosa
from tqdm import tqdm

from data import load_vocab, load_train_data
from data_load import get_batch
from hyperparams import Hyperparams as hp
from modules import *
from networks import encode, decode
import numpy as np
import tensorflow as tf
from utils import shift_by_one

token2idx, idx2token = load_vocab()
 
class Graph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch() # 
            else: # Evaluation
                self.x = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            
            self.decoder_inputs = embed(shift_by_one(self.y), len(token2idx), hp.embed_size) # (N, T, E)
            
            # Encoder
            self.memory = encode(self.x, is_training=is_training) # (N, T, E)
             
            # Decoder
            self.outputs = decode(self.decoder_inputs, self.memory) # (N, T', hp.n_mels*hp.r)
             
            if is_training:  
                # Loss
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
                
                # Target masking
                if hp.target_zeros_masking:
                    self.istarget = tf.to_float(tf.not_equal(self.y, 0))
                    self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget) + 1e-7)
                
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
         
def main():   
    g = Graph(); print("Training Graph loaded")
    
    with g.graph.as_default():
        # Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)
                    
                    if step % 10 == 0:
                        print(sess.run(g.mean_loss))
                # Write checkpoint files at every epoch
                l, gs = sess.run([g.mean_loss, g.global_step])
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d_%.2f' % (epoch, gs, l))

if __name__ == '__main__':
    main()
    print("Done")
