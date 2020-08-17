from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac

"""You can add your own configuration function to this file and select
it using `--proto function_name`."""


def base_config():
    config = {}

    ### Locations of input/output files

    # The name of the model
    config['model_name']        = 'model_name'

    # Source and target languages
    # Input files should be named with these as extensions
    config['src_lang']          = 'src_lang'
    config['trg_lang']          = 'trg_lang'

    # Directory to read input files from
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])

    # Directory to save models and other outputs in
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])

    # Pathname of log file
    config['log_file']          = './nmt/DEBUG.log'

    ### Model options

    # Filter out sentences longer than this (minus one for bos/eos)
    config['max_train_length']  = 1000

    # Vocabulary sizes
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['share_vocab']       = False

    # Normalize word embeddings (Nguyen and Chiang, 2018)
    config['fix_norm']          = True

    # Tie word embeddings
    config['tie_mode']          = ac.ALL_TIED

    # Whether to learn position encodings
    config['learned_pos']       = False
    # Position encoding size
    config['max_pos_length']    = 1024
    
    # Layer sizes
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8

    # Whether residual connections should bypass layer normalization
    # if True, layer-norm->dropout->add
    # if False, dropout->add->layer-norm (as in original paper)
    config['norm_in']           = True

    ### Dropout/smoothing options

    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['label_smoothing']   = 0.1

    ### Training options

    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['normalize_loss']    = ac.LOSS_TOK

    # Hyperparameters for Adam optimizer
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8

    # Learning rate
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3

    # Gradient clipping
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9

    ### Validation/stopping options

    config['max_epochs']        = 20
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True

    # Undo BPE segmentation when validating
    config['restore_segments']  = True

    # How many of the best models to save
    config['n_best']            = 1

    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    ### Decoding options

    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['dict_window']       = 9999

    config['dict_mode']         = 'src_to_src'   
    config['dict_add_method']   = 'append'
    config['skip']		        = False  
    config['dict_window']       = 50
    config['dict_learned_pos']  = False  
    config['apply_dict_all']    = False
    config['concate_pos']       = False
    config['split_pos_encoding']= False
    config['thres']             = 10

    return config


def Laws_word():
    config=base_config()
    config['model_name']        = 'Laws_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 22000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Laws_word_bpe():
    config=Laws_word()
    config['model_name']        = 'Laws_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')   
    return config 

def News_word():
    config=base_config()
    config['model_name']        = 'News_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 24000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config

def News_word_bpe():
    config=News_word()
    config['model_name']        = 'News_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')   
    return config


def News_mult_bpe():
    config=base_config()
    config['model_name']        = 'News_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 24000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config


def Laws_mult_bpe():
    config=base_config()
    config['model_name']        = 'Laws_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 22000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config

def Education_word():
    config=base_config()
    config['model_name']        = 'Education_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 28000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Education_word_bpe():
    config=Education_word()
    config['model_name']        = 'Education_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')    
    return config 


def Education_mult_bpe():
    config=base_config()
    config['model_name']        = 'Education_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 28000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config

def Science_word():
    config=base_config()
    config['model_name']        = 'Science_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 27000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Science_word_bpe():
    config=Science_word()
    config['model_name']        = 'Science_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config

def Science_mult_word():
    config=base_config()
    config['model_name']        = 'Science_mult_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 27000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Science_word_cihai():
    config=Science_word()
    config['model_name']        = 'Science_word_cihai'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cihai_word.ent'
    config['src_dict_def']      = 'cihai_word.def'   
    return config 

def Science_word_comb():
    config=Science_word()
    config['model_name']        = 'Science_word_comb'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cihai_cedict.ent'
    config['src_dict_def']      = 'cihai_cedict.def'   
    return config 


def Science_mult_bpe():
    config=base_config()
    config['model_name']        = 'Science_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 27000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config

def Science_mult_bpe_cihai():
    config=Science_mult_bpe()
    config['model_name']        = 'Science_mult_bpe_cihai'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cihai.nos'
    config['src_dict_def']      = 'cihai.def'   
    return config

def Science_mult_bpe_comb():
    config=Science_mult_bpe()
    config['model_name']        = 'Science_mult_bpe_comb'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cihai_cedict.nos'
    config['src_dict_def']      = 'cihai_cedict.def'   
    return config


def Spoken_word():
    config=base_config()
    config['model_name']        = 'Spoken_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 25000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Spoken_word_cihai():
    config=Spoken_word()
    config['model_name']        = 'Spoken_word_cihai'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cihai_word.ent'
    config['src_dict_def']      = 'cihai_word.def'   
    return config


def Spoken_word_comb():
    config=Spoken_word()
    config['model_name']        = 'Spoken_word_comb'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cihai_cedict.ent'
    config['src_dict_def']      = 'cihai_cedict.def'   
    return config

def Spoken_word_all():
    config=base_config()
    config['model_name']        = 'Spoken_word_all'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 25000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True   
    config['apply_dict_all']    = True  
    return config 

def Spoken_mult_word():
    config=base_config()
    config['model_name']        = 'Spoken_mult_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 25000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 


def Spoken_mult_bpe():
    config=base_config()
    config['model_name']        = 'Spoken_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 25000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Spoken_mult_bpe_all():
    config=base_config()
    config['model_name']        = 'Spoken_mult_bpe_all'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 25000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True    
    config['apply_dict_all']    = True 
    return config 

def Spoken_mult_bpe_cihai():
    config=Spoken_mult_bpe()
    config['model_name']        = 'Spoken_mult_bpe_cihai'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt') 
    config['src_dict_ent']      = 'cihai.nos'
    config['src_dict_def']      = 'cihai.def'   
    return config 


def Spoken_word_bpe():
    config=base_config()
    config['model_name']        = 'Spoken_word_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 25000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Subtitles_word():
    config=base_config()
    config['model_name']        = 'Subtitles_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 27000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Subtitles_word_bpe():
    config=Subtitles_word()
    config['model_name']        = 'Subtitles_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')    
    return config


def Subtitles_mult_bpe():
    config=base_config()
    config['model_name']        = 'Subtitles_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 27000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Thesis_word():
    config=base_config()
    config['model_name']        = 'Thesis_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 27000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def Thesis_word_bpe():
    config=Thesis_word()
    config['model_name']        = 'Thesis_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')    
    return config 


def Thesis_mult_bpe():
    config=base_config()
    config['model_name']        = 'Thesis_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 27000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 


def UM_all_word():
    config=base_config()
    config['model_name']        = 'UM_all_word'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 33000
    config['max_epochs']        = 10
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.ent'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def UM_all_word_bpe():
    config=UM_all_word()
    config['model_name']        = 'UM_all_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')   
    return config


def UM_all_mult_bpe():
    config=base_config()
    config['model_name']        = 'UM_all_mult_bpe'
    config['src_lang']          = 'ch'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 33000
    config['max_epochs']        = 10
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'cedict.nos'
    config['src_dict_def']      = 'cedict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 


def PARL_deu_word():
    config=base_config()
    config['model_name']        = 'PARL_deu_word'
    config['src_lang']          = 'de'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 16000
    config['max_epochs']        = 10
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'freedict_deu-eng.ent'
    config['src_dict_def']      = 'freedict_deu-eng.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def PARL_deu_word_bpe():
    config=PARL_deu_word()
    config['model_name']        = 'PARL_deu_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')    
    return config 


def PARL_deu_mult_bpe():
    config=base_config()
    config['model_name']        = 'PARL_deu_mult_bpe'
    config['src_lang']          = 'de'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 16000
    config['max_epochs']        = 10
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'freedict_deu-eng.nos'
    config['src_dict_def']      = 'freedict_deu-eng.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 



def PARL_small_word():
    config=base_config()
    config['model_name']        = 'PARL_small_word'
    config['src_lang']          = 'de'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 16000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'freedict_deu-eng.ent'
    config['src_dict_def']      = 'freedict_deu-eng.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def PARL_small_word_bpe():
    config=PARL_small_word()
    config['model_name']        = 'PARL_small_word_bpe'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config 


def PARL_small_mult_word():
    config=PARL_small_word()
    config['model_name']        = 'PARL_small_mult_word'
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt') 
    config['src_dict_ent']      = 'freedict_deu-eng.nos'  
    return config


def PARL_small_mult_bpe():
    config=base_config()
    config['model_name']        = 'PARL_small_mult_bpe'
    config['src_lang']          = 'de'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 16000
    config['max_epochs']        = 20
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'freedict_deu-eng.nos'
    config['src_dict_def']      = 'freedict_deu-eng.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 


def ta2en_word():
    config=base_config()
    config['model_name']        = 'ta2en_word'
    config['src_lang']          = 'ta'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 8000
    config['max_epochs']        = 100
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'ta2en_dict.ent'
    config['src_dict_def']      = 'ta2en_dict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def ta2en_mult_bpe():
    config=base_config()
    config['model_name']        = 'ta2en_mult_bpe'
    config['src_lang']          = 'ta'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 8000
    config['max_epochs']        = 100
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'ta2en_dict.nos'
    config['src_dict_def']      = 'ta2en_dict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 

def ta2en_bpe():
    config=base_config()
    config['model_name']        = 'ta2en_bpe'
    config['src_lang']          = 'ta'
    config['trg_lang']          = 'en'
    config['joint_vocab_size']  = 0
    config['max_epochs']        = 100
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['log_file']          = './nmt/saved_models/{}/DEBUG.log'.format(config['model_name'])
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['src_dict_ent']      = 'ta2en_dict.ent'
    config['src_dict_def']      = 'ta2en_dict.def'
    config['learned_pos']       = False
    config['dict_learned_pos']  = True     
    return config 


