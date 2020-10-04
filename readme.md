# Attach-Dictionary Extension of Witwicky: An implementation of Transformer in PyTorch

[Xing Jie Zhong](), University of Notre Dame

Witwicky Base for this comes from [Toan Q. Nguyen](http://tnq177.github.io), University of Notre Dame

An extension of Witwicky https://github.com/tnq177/witwicky (please consult link for general instructions) to read in and attach defintions from a dictionary to the encoder. 

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
The options that are added in this version (not found in witwicky) include:

``skip''
``dict_window'' 
``dict_learned_pos''
``apply_dict_all''
``concate_pos''
``split_pos_encoding''
``thres''

## References

Parts of code/scripts are borrowed/inspired from:  

* https://github.com/pytorch/fairseq
* https://github.com/tensorflow/tensor2tensor
* https://github.com/EdinburghNLP/nematus/
* https://github.com/mila-iqia/blocks
* https://github.com/moses-smt/mosesdecoder
