import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from layers import Encoder, Decoder
import nmt.all_constants as ac
import nmt.utils as ut
from os.path import join

class Model(nn.Module):
    """Model"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.init_embeddings()
        self.init_model()

    def init_external_embeddings(self):
        if(self.config['pretrain']!='N/A'):
            print('importing embeddings')
            import gensim
            import os
            self.ground_emb = gensim.models.KeyedVectors.load_word2vec_format(self.config['pretrain'])
            self.dictword={}
            self.wordlist=[]
            for line in open(os.path.join(self.config['data_dir'],'joint_vocab.'+self.config['src_lang']+'-'+self.config['trg_lang'])):
                word, key, _=line.strip('\n').split()
                self.dictword[key]=word 
                self.wordlist.append(key)
            if(self.config['pretrain2']!='N/A'):
                print('importing second word embeddings')
                self.ground_emb2 = gensim.models.KeyedVectors.load_word2vec_format(self.config['pretrain2'])
        if(self.config['Out_pretrain']!='N/A'):
            print('Importing output embeddings')
            import gensim
            self.out_pretrain_embs=gensim.models.KeyedVectors.load_word2vec_format(self.config['Out_pretrain'])

    def load_dictionary(self):
        self.dict={}
        ent_path=join(self.config['data_dir'],self.config['src_dict_ent'])
        def_path=join(self.config['data_dir'],self.config['src_dict_def'])
        lastent=''
        for ents, defs in zip(open(ent_path),open(def_path)):
            entry=ents.strip('\n')
            defini=defs.strip('\n')
            if(entry==lastent):#middle entry or last entry
                if(entry in self.dict):
                    self.dict[entry]=self.dict[entry]+' '+defini
                else:
                    raise Exception('Something wrong with loading dictionary')
            else:
                self.dict[entry]=defini
            lastent=entry            
            self.dict[ents.strip('\n')]=defs.strip('\n')
                    
        self.dict_max_len=0
        for item in self.dict:
            self.dict_max_len=max(len(self.dict[item].split()),self.dict_max_len)
        print(len(self.dict),self.dict_max_len)
        self.dict_max_len=min(self.config['dict_window'],self.dict_max_len) 

    def load_joint_vocab(self):
        self.tag_to_word={}
        self.word_to_tag={}
        vocabfile='joint_vocab.{}-{}'.format(self.config['src_lang'],self.config['trg_lang'])
        for line in open(join(self.config['data_dir'],vocabfile)):
            word, tag, freq= line.strip('\n').split()
            self.tag_to_word[tag]=word
            self.word_to_tag[word]=tag        
                
    def init_embeddings(self):
        embed_dim = self.config['embed_dim']
        tie_mode = self.config['tie_mode']
        fix_norm = self.config['fix_norm']
        max_pos_length = self.config['max_pos_length']
        learned_pos = self.config['learned_pos']
        
        self.init_external_embeddings()
        self.load_joint_vocab()
        self.load_dictionary()
        
        # get positonal embedding
        if not learned_pos:
            self.pos_embedding = ut.get_positional_encoding(embed_dim, max_pos_length)
            self.dict_pos_embedding = ut.get_positional_encoding(embed_dim, self.dict_max_len)
        else:
            self.pos_embedding = Parameter(torch.Tensor(max_pos_length, embed_dim))
            self.dict_pos_embedding = Parameter(torch.Tensor(self.dict_max_len, embed_dim))
            nn.init.normal_(self.pos_embedding, mean=0, std=embed_dim ** -0.5)
            nn.init.normal_(self.dict_pos_embedding, mean=0, std=embed_dim ** -0.5)


        # get word embeddings
        src_vocab_size, trg_vocab_size = ut.get_vocab_sizes(self.config)
        self.src_vocab_mask, self.trg_vocab_mask = ut.get_vocab_masks(self.config, src_vocab_size, trg_vocab_size)
        if tie_mode == ac.ALL_TIED:
            src_vocab_size = trg_vocab_size = self.trg_vocab_mask.shape[0]

        self.out_bias = Parameter(torch.Tensor(trg_vocab_size))
        nn.init.constant_(self.out_bias, 0.)

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.out_embedding = self.trg_embedding.weight
        self.embed_scale = embed_dim ** 0.5
               


        self.FT_UNK= torch.zeros(300,dtype=torch.float32,requires_grad=True)
        nn.init.uniform_(self.FT_UNK,a=-0.01,b=0.01)
        self.FT_UNK=Parameter(self.FT_UNK) 
        self.transform_wordemb= nn.Linear(300,embed_dim,bias=False)
        if(self.config['init_embtrans']== 'xavier'):
            nn.init.xavier_uniform_(self.transform_wordemb.weight, gain=1.0)             
        else:
            ranga,rangb=self.config['init_embtrans']
            nn.init.uniform_(self.transform_wordemb.weight,a=ranga,b=rangb)
        
        if(self.config['bpe_pos']=='add'):                    
            self.BPE_Head= torch.zeros(embed_dim,dtype=torch.float32,requires_grad=True)
            self.BPE_Mid= torch.zeros(embed_dim,dtype=torch.float32,requires_grad=True)   
            self.BPE_End= torch.zeros(embed_dim,dtype=torch.float32,requires_grad=True)  
            self.BPE_NONE= torch.zeros(embed_dim,dtype=torch.float32,requires_grad=True)      
            nn.init.uniform_(self.BPE_Head,a=-0.01,b=0.01)
            nn.init.uniform_(self.BPE_Mid,a=-0.01,b=0.01)
            nn.init.uniform_(self.BPE_End,a=-0.01,b=0.01)
            nn.init.uniform_(self.BPE_NONE,a=-0.01,b=0.01)             
            self.BPE_Head=Parameter(self.BPE_Head)  
            self.BPE_Mid=Parameter(self.BPE_Mid)  
            self.BPE_End=Parameter(self.BPE_End) 
            self.BPE_NONE=Parameter(self.BPE_NONE) 
        if(self.config['we_decoder']):
            ranga,rangb=self.config['init_embtrans']
            self.TO_WE=nn.Linear(embed_dim,300,bias=False)
            nn.init.uniform_(self.TO_WE.weight,a=ranga,b=rangb)
            
        elif(self.config['bpe_pos']=='transform'):
            
            self.BPE_Head_transform=nn.Linear(300,embed_dim,bias=False)
            self.BPE_Mid_transform=nn.Linear(300,embed_dim,bias=False)    
            self.BPE_End_transform=nn.Linear(300,embed_dim,bias=False) 
            nn.init.uniform_(self.BPE_Head_transform.weight,a=ranga,b=rangb)
            nn.init.uniform_(self.BPE_Mid_transform.weight,a=ranga,b=rangb)
            nn.init.uniform_(self.BPE_End_transform.weight,a=ranga,b=rangb)
            
        
        if tie_mode == ac.ALL_TIED:
            self.src_embedding.weight = self.trg_embedding.weight

        if not fix_norm:
            nn.init.normal_(self.src_embedding.weight, mean=0, std=embed_dim ** -0.5)
            nn.init.normal_(self.trg_embedding.weight, mean=0, std=embed_dim ** -0.5)
        else:
            d = 0.01 # pure magic
            nn.init.uniform_(self.src_embedding.weight, a=-d, b=d)
            nn.init.uniform_(self.trg_embedding.weight, a=-d, b=d)
            

    def init_model(self):
        num_enc_layers = self.config['num_enc_layers']
        num_enc_heads = self.config['num_enc_heads']
        num_dec_layers = self.config['num_dec_layers']
        num_dec_heads = self.config['num_dec_heads']

        embed_dim = self.config['embed_dim']
        ff_dim = self.config['ff_dim']
        dropout = self.config['dropout']
        norm_in = self.config['norm_in']

        # get encoder, decoder
        self.encoder = Encoder(num_enc_layers, num_enc_heads, embed_dim, ff_dim, dropout=dropout, norm_in=norm_in)
        self.decoder = Decoder(num_dec_layers, num_dec_heads, embed_dim, ff_dim, dropout=dropout, norm_in=norm_in)

        # leave layer norm alone
        init_func = nn.init.xavier_normal_ if self.config['weight_init_type'] == ac.XAVIER_NORMAL else nn.init.xavier_uniform_
        for m in [self.encoder.self_atts, self.encoder.pos_ffs, self.decoder.self_atts, self.decoder.pos_ffs, self.decoder.enc_dec_atts]:
            for p in m.parameters():
                if p.dim() > 1:
                    init_func(p)
                else:
                    nn.init.constant_(p, 0.)

    def get_input(self, toks, is_src=True):
        embeds = self.src_embedding if is_src else self.trg_embedding
        word_embeds = embeds(toks) # [bsz, max_len, embed_dim]
        if self.config['fix_norm']:
            word_embeds = ut.normalize(word_embeds, scale=False)
        else:
            word_embeds = word_embeds * self.embed_scale

        pos_embeds = self.pos_embedding[:toks.size()[-1], :].unsqueeze(0) # [1, max_len, embed_dim]
        return word_embeds + pos_embeds

    def split_tokword(self,intok,device):
        if(len(str(intok))>1):
            if('_' in intok):
                intok,word,other=intok.split('_',2)
                if(other[0]=='_'):
                    def_word='_'
                    def_pos=other[2:]
                else:
                    def_word,def_pos=other.split('_',1)                
                intok=torch.tensor(int(intok)).to(device)
            else:
                intok=torch.tensor(int(intok)).to(device)
                word=''  
                def_word=''
                def_pos=''
        else:#paddings BOS EOS case
            intok=torch.tensor(int(intok)).to(device)
            word=''
            def_word=''
            def_pos=''                 
        return (intok, word, def_word, def_pos)       
        
    def determine_pos(self,i,intok,word_pos):
        if(len(word_pos)>0):            
            sent_pos_word=self.pos_embedding[int(word_pos)]                    
        elif(i<1024 and intok.item()==0):
            sent_pos_word=self.pos_embedding[i]
        elif(i>1023 and intok.item()==0):
            sent_pos_word=self.pos_embedding[1023]
        else:
            raise Exception('Problem here with word positional embeddings')
        return sent_pos_word
    
    
    def determine_word_emb(self,intok,def_word,def_pos,embeds,plain_sent,device):
        if(def_pos=='NA' or len(def_pos)<1):
            return (self.norm_we(embeds(intok)),self.combine_tensor(intok,plain_sent))
            
        elif(self.config['skip']):
            return (self.norm_we(embeds(intok)),plain_sent)
        
        elif(def_pos !='NA'):
            #print(intok,def_word,def_pos)
            if(def_word in self.word_to_tag):
                #print('def_word known is:',def_word) 
                def_tok=torch.tensor(int(self.word_to_tag[def_word])).to(device)
            else:
                #print('def_word unkown is:',def_word)
                def_tok=torch.tensor(3).to(device)    
            return (self.norm_we(embeds(def_tok))+self.dict_pos_embedding[int(def_pos)],plain_sent)   
        

    def combine_tensor(self,inten,outten):
        if(len(outten)<1):
            outten = inten.unsqueeze(0)
        else:
            outten = torch.cat((outten,inten.unsqueeze(0)),0)#[max_len, embed_dim]
        return outten  
        
    def resize_maxlen(self,word_embeds,new_max_len,pad_emb,dimen):
        old_len=word_embeds.size()[dimen]   
        mults= new_max_len - old_len
        if(mults<0):
            print('resizing:',mults,new_max_len,word_embeds)
            raise Exception('max len is wrong in resizing')
        elif mults<1:
            return word_embeds
        appends_sent=torch.cat([pad_emb]*mults)
        if(dimen==0):
            #print(word_embeds, appends_sent)
            new_embeds=torch.cat((word_embeds,appends_sent),0)
        elif(dimen==1):
            new_embeds=[]
            #print('appendings dim:',appends_sent.size())
            for batch in word_embeds:
                batch_emb=torch.cat((batch,appends_sent),0)
                new_embeds=self.combine_tensor(batch_emb,new_embeds)     
        return new_embeds    

    def get_input2(self, toks_str, toks_cuda, device, is_src):
        embeds = self.src_embedding if is_src else self.trg_embedding
        unkcount=0
        none_count=0
        plain_embed=[]
        word_embed=[]
        pos_embed=[]
        embed_dim=self.config['embed_dim']
        for tok in toks_str:
            sent=[]
            sent_pos=[]
            for i, intoki in enumerate(tok):
                intok, word_pos, def_word, def_pos = self.split_tokword(intoki,device)
                if(len(word_pos)>0):
                    if(def_pos=='NA'):
                        sent_pos_word=self.pos_embedding[int(word_pos)]
                    #else:
                            
                elif(i<1024 and intok.item()==0):
                    sent_pos_word=self.pos_embedding[i]
                elif(i>1023 and intok.item()==0):
                    sent_pos_word=self.pos_embedding[1023]
                else:
                    print(intok,word_pos, def_word, def_pos,intoki,tok)
                    raise Exception('Problem here',intok,i,word_pos)
                    
                sent_pos=self.combine_tensor(sent_pos_word,sent_pos)                
                sent=self.combine_tensor(embeds(intok),sent)
            word_embed=self.combine_tensor(sent,word_embed)
            pos_embed=self.combine_tensor(sent_pos,pos_embed)         
        word_embeds = word_embed.to(device)
        pos_embeds = pos_embed.to(device)
        word_embeds=self.norm_we(word_embeds)
        return word_embeds + pos_embeds 
                
    def norm_we(self,word_embeds):
        if self.config['fix_norm']:
            word_embeds = ut.normalize(word_embeds, scale=False)
        else:
            word_embeds = word_embeds * self.embed_scale  
        return word_embeds     

    def get_input3(self, toks_str, toks_cuda, device, is_src):
        embeds = self.src_embedding if is_src else self.trg_embedding
        plain_embed=[]
        pad_emb=embeds(torch.tensor([0]).to(device))
        pads=torch.tensor([0]).to(device)
        word_embed=[]
        new_max_len=0 

        for tok in toks_str:
            sent=[]
            plain_sent=[]
            for i, intoki in enumerate(tok):
                intok, word_pos, def_word, def_pos = self.split_tokword(intoki,device)
                sent_pos_word= self.determine_pos(i,intok,word_pos)
                word_embi,plain_sent=self.determine_word_emb(intok,def_word,def_pos,embeds,plain_sent,device)                         
                sent=self.combine_tensor(word_embi+sent_pos_word,sent)
                
            #if(len(plain_embed)>0):
                #if(plain_embed.size()[1]!=plain_sent.size()[0]):
                    #print(new_max_len,plain_embed.size(),plain_sent.size(),'\n',plain_embed, plain_sent) 
                    
            if(len(plain_embed)>0 and new_max_len!=len(plain_sent)):
                new_max_len=max(len(plain_sent),plain_embed.size()[1])
                #print('before resizing:',plain_embed,plain_sent)
                plain_embed = self.resize_maxlen(plain_embed,new_max_len,pads,1)
                plain_sent = self.resize_maxlen(plain_sent,new_max_len,pads,0)
                #print('after resizing:',plain_embed.size(),plain_sent.size())           
            plain_embed=self.combine_tensor(plain_sent,plain_embed)    
            word_embed=self.combine_tensor(sent,word_embed)       
        word_embeds = word_embed.to(device)

        return (word_embeds,plain_embed)        
                                                    

    def forward(self, src_toks_cpu, src_toks, trg_toks_cpu, trg_toks, targets, device):
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        decoder_mask = torch.triu(torch.ones((trg_toks.size()[-1], trg_toks.size()[-1])), diagonal=1).type(trg_toks.type()) == 1
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1)
        
        unkcount_enc=0
        unkcount_dec=0
        none_count_enc=0
        none_count_dec=0

        #encoder_inputs, unkcount_enc, none_count_enc = self.get_embinput2(src_toks_cpu, src_toks, device, is_src=True)
        #encoder_inputs = self.get_input(src_toks, is_src=True)
        encoder_inputs,_ = self.get_input3(src_toks_cpu, src_toks,device, is_src=True)
        decoder_inputs = self.get_input(trg_toks, is_src=False)        
       
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)
        decoder_outputs = self.decoder(decoder_inputs, decoder_mask, encoder_outputs, encoder_mask)
        logits = self.logit_fn(decoder_outputs,False,True,trg_toks_cpu)
        neglprobs = F.log_softmax(logits, -1)
        neglprobs = neglprobs * self.trg_vocab_mask.type(neglprobs.type()).reshape(1, -1)
        targets = targets.reshape(-1, 1)
        non_pad_mask = targets != ac.PAD_ID
        nll_loss = -neglprobs.gather(dim=-1, index=targets)[non_pad_mask]
        smooth_loss = -neglprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        label_smoothing = self.config['label_smoothing']

        if label_smoothing > 0:
            loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / self.trg_vocab_mask.type(smooth_loss.type()).sum()
        else:
            loss = nll_loss

        return {
            'loss': loss,
            'nll_loss': nll_loss,
            'unkcount_enc':unkcount_enc,
            'unkcount_dec':unkcount_dec,
            'none_count_enc':none_count_enc,
            'none_count_dec':none_count_dec
            
        }

    def logit_fn(self, decoder_output,from_beam=False,ref_toks=False,trg_toks=''):
        softmax_weight = self.out_embedding if not self.config['fix_norm'] else ut.normalize(self.out_embedding, scale=True) 
        logits = F.linear(decoder_output, softmax_weight, bias=self.out_bias)
        logits = logits.reshape(-1, logits.size()[-1])
        logits[:, ~self.trg_vocab_mask] = -1e9
        return logits

    def beam_decode(self, src_toks_cpu, src_toks, device='cuda:0'):
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        
        #encoder_inputs, unkcount_enc, none_count_enc = self.get_embinput2(src_toks_cpu, src_toks, device, is_src=True) 
        encoder_inputs, plan_emb = self.get_input3(src_toks_cpu, src_toks,device, is_src=True)
        #encoder_inputs = self.get_input(src_toks, is_src=True)          
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)
        #max_lengths = torch.sum(src_toks != ac.PAD_ID, dim=-1).type(src_toks.type()) + 50 
        print('Old max len size is: ',src_toks.size(),' new max len size is: ',plan_emb.size())
        max_lengths = torch.sum(plan_emb != ac.PAD_ID, dim=-1).type(src_toks.type()) + 50 
            
        def get_trg_inp(ids, time_step):
            ids = ids.type(src_toks.type())
            word_embeds = self.trg_embedding(ids)
            if self.config['fix_norm']:
                word_embeds = ut.normalize(word_embeds, scale=False)
            else:
                word_embeds = word_embeds * self.embed_scale

            pos_embeds = self.pos_embedding[time_step, :].reshape(1, 1, -1)
            return word_embeds + pos_embeds

        def logprob(decoder_output):
            #print('From logprob\n\n')
            return F.log_softmax(self.logit_fn(decoder_output,from_beam=True), dim=-1)

        return self.decoder.beam_decode(encoder_outputs, encoder_mask, get_trg_inp, logprob, ac.BOS_ID, ac.EOS_ID, max_lengths, beam_size=self.config['beam_size'], alpha=self.config['beam_alpha'])


