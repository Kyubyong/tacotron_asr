# Speech Recognition Using Tacotron


## Motivation
Tacotron is an end-to-end speech generation model which was first introduced in [Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135). It takes as input text at the character level, and targets mel filterbanks and the linear spectrogram. Although it is a generation model, I felt like testing how well it can be applied to the speech recognition task.

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow == 1.1
  * librosa

## Data
I use the [VCTK Corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html), one of the most popular speech corpora, for my experiment. Because there's no pre-defined split of training and evaluation, 10*(mini batch) samples that don't appear in the training set are reserved for evaluation.

## Contents
  * `hyperparams.py` includes all hyper parameters.
  * `prepro.py` creates training and evaluation data to `data/` folder.
  * `data_load.py` loads data and put them in queues so multiple mini-bach data are generated in parallel.
  * `utils.py` has some operational functions.
  * `modules.py` contains building blocks for encoding and decoding networks.
  * `networks.py` defines encoding and decoding networks.
  * `train.py` executes training.
  * `eval.py` executes evaluation.

## Training
  * STEP 1. Download and extract [VCTK Corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) and adjust the value of 'vctk' in `hyperparams.py`.
  * STEP 2. Adjust other hyper parameters in `hyperparams.py` if necessary.
  * STEP 3. Run `train_multiple_gpus.py` if you want to use more than one gpu, otherwise `train.py`.

## Evaluation
  * Run `eval.py` to get speech recognition results for the test set.

## Related projects
  * [A TensorFlow Implementation of Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model](https://github.com/Kyubyong/tacotron)
  * [Speech-to-Text-WaveNet : End-to-end sentence level English speech recognition based on DeepMind's WaveNet and tensorflow](https://github.com/buriburisuri/speech-to-text-wavenet)

