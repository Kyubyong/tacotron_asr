# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron_asr
'''

class Hyperparams:
    '''Hyper parameters'''
    # data
    vctk = 'VCTK-Corpus/txt/*/*.txt'
    web = 'WEB'
    use_vctk = False
    use_web = True
    max_len = 100 # maximum length of text
    max_duration = 5.0 # maximum duration of a sound file. seconds.
    
    # signal processing
    sr = 22050 # Sampling rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    
    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    
    # training scheme
    lr = 0.00001 
    logdir = "logdir"
    batch_size = 32
    num_epochs = 300 
    
    # inference
    beam_width = 1 # if beam width is 1, we apply a regular greedy decoding.
