# Attach-Dictionary Extension of Witwicky: An implementation of Transformer in PyTorch

[Xing Jie Zhong](), University of Notre Dame

[Toan Q. Nguyen](http://tnq177.github.io), University of Notre Dame

An implementation of Vaswani et al.'s [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) with [PyTorch](https://pytorch.org). An [early version](https://github.com/tnq177/nmt_text_from_non_native_speaker) of this code was used in our paper [Neural Machine Translation of Text from Non-Native Speakers
](https://arxiv.org/abs/1808.06267).  

This code has been tested with only Python 3.6 and PyTorch 1.0.

## Input and Preprocessing

The code expects bitext data with filenames

    train.src_lang   train.trg_lang
    dev.src_lang     dev.trg_lang
    test.src_lang    test.trg_lang
    headwords.ent    definitions.def

The following data preprocessing should happen first:  
* clean dictionary data and seperate the headwords and definitions into two files
* The dictionary can be either src_lang to src_lang or src_lang to trg_lang but not trg_lang to trg_lang
* tokenize and/or segment all data excluding the dictionary headword file
* learn BPE from training data
* apply BPE to all files excluding the dictionary headword file

## Usage

To train a new model:  
* Write a new configuration function in ``configurations.py``  
* Put preprocessed data in ``nmt/data/model_name`` or as configured in ``data_dir`` option in your configuration function  
* Put the name of the processed dictionary headword file in ``src_dict_ent`` and dictionary definition file in ``src_dict_def``
* Run: ``python3 -m nmt --proto config_name``  

During training, the model is validated on the dev set, and the best checkpoint is saved to ``nmt/saved_models/model_name/model_name-SCORE.pth``.

The `val_by_bleu` option controls whether the best checkpoint is chosen based on dev BLEU score (`val_by_bleu=True`) or (label-smoothed) dev perplexity (`val_by_bleu=False`).

The ``n_best`` option tells the trainer to save the best `n_best` checkpoints; however, it's a bug that because of the way the checkpoints are named, if two checkpoints happen to have the same score, the earlier one is overwritten.

When training is finished, the best checkpoint is reloaded and used for decoding the test file. To decode another file, run ``python3 -m nmt --proto config_name --model-file path_to_checkpoint --input-file path_to_file_to_decode``.  

We support minibatched beam search, but currently it's quite ad-hoc. We assume that during training, if a minibatch size of ``batch_size`` doesn't run out of memory, during beam search we can use a minibatch size of ``batch_size//beam_size`` (see ``get_trans_input`` function in ``data_manager.py``).

## Options

Options are set in `configurations.py`. Many are pretty important.



## Recommendations

### General


### How long to train

It's common to train Transformer for about 100k iterations. This works out to be around 4-50 epochs for Arabic-English. However, we can see that from epoch 50 to 100 we can still get some good gain. Note that for all models here, we use a minibatch size of 4096 tokens instead of 25k tokens. My general rule of thumb is for datasets of around 50k-500k examples, we should train around 100 epochs. Coming from LSTM, Transformer is so fast that training a bit longer still doesn't seem to take much time. See table below for some stats:  

|                                         | ar2en | de2en | he2en | it2en |
|-----------------------------------------|-------|-------|-------|-------|
| # train examples                        | 212k  | 166k  | 210k  | 202k  |
| # dev examples                          | 4714  | 4148  | 4515  | 4547  |
| # test examples                         | 5953  | 4491  | 5508  | 5625  |
| # train target tokens                   | 5.1M  | 3.8M  | 5M    | 4.7M  |
| Training speed (# target tokens/second) | 10.2k | 9.2k  | 10.2k | 9.3k  |
| Total time for 100 epochs (hours)       | ~19   | ~16   | ~18   | ~20   |    




I'm pretty surprised we got much better BLEU than the multilingual baseline. Note that all of my baselines are bilingual only.

|                                                                                                      | en2vi              | ar2en | de2en | he2en | it2en | KFTT en2ja           |
|------------------------------------------------------------------------------------------------------|--------------------|-------|-------|-------|-------|----------------------|
| [Massively Multilingual NMT-baseline](https://arxiv.org/abs/1903.00089)                              | ---                | 27.84 | 30.5  | 34.37 | 33.64 | ---                  |
| [Massively Multilingual NMT-multilingual](https://arxiv.org/abs/1903.00089)                          | ---                | 28.32 | 32.97 | 33.18 | 35.14 | ---                  |
| [SwitchOut](https://arxiv.org/pdf/1808.07512.pdf), word-based, transformer                           | 29.09              | ---   | ---   | ---   | ---   | ---                  |
| [duyvuleo's transformer dynet](https://github.com/duyvuleo/Transformer-DyNet), transformer, ensemble | 29.71 (word-based) | ---   | ---   | ---   | ---   | 26.55 (BPE+ensemble) |
| [Nguyen and Chiang](https://aclweb.org/anthology/N18-1031), LSTM, word-based                         | 27.5               | ---   | ---   | ---   | ---   | 26.2                 |
| this-code (BPE)                                                                                      | 31.71              | 33.15 | 37.83 | 38.79 | 40.22 | ---                  |
| this-code + fixnorm (BPE)                                                                            | 31.77              | 33.39 | 38.15 | 39.08 | 40.33 | ---                  |
| this-code, word-based                                                                                | 29.47 (4layers)    | ---   | ---   | ---   | ---   | 31.28 (6layers)      |

## References

Parts of code/scripts are borrowed/inspired from:  

* https://github.com/pytorch/fairseq
* https://github.com/tensorflow/tensor2tensor
* https://github.com/EdinburghNLP/nematus/
* https://github.com/mila-iqia/blocks
* https://github.com/moses-smt/mosesdecoder
