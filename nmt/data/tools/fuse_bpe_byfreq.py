from collections import Counter

               
def load_dictionary(ent_path,def_path):
    dicti={}
    for ents, defs in zip(open(ent_path),open(def_path)):
        entry=ents.strip()
        defini=defs.strip()
        if(entry!=defini):
            dicti[entry]=defini
    sorted_items= sorted(dicti.items(), key=lambda item: len(item[1]), reverse=True)
    out=dict(sorted_items) 
    return out                   

    
def refine_revdict(train,dic,thres):
    counts=Counter()
    rem=0
    print('Pre refining dictionary length: ',len(dic))
    for line in open(train):
        templine=line.strip()
        for seq in dic:
            if seq in templine:
                counts[seq]+=1
    for seq in list(dic):
        if(seq not in counts):
            del dic[seq]
        elif(counts[seq]>thres):
            del dic[seq]
            rem+=1
    print('There are ',rem,' words that fave been fused ',thres,' times or more') 
    print('Post refining dictionary length: ',len(dic))   
    sorted_items= sorted(dic.items(), key=lambda item: len(item[1]), reverse=True)  
    out=dict(sorted_items)
    return out, counts  

def load_counts(train,dic):
    counts=Counter() 
    for line in open(train):
        templine=line.strip()
        for seq in dic:
            if seq in templine:
                counts[seq]+=1
    return counts                        

        
def rewrite(infile, outfile, dic, counts,thres): 
    outs=open(outfile,'w')
    count=0
    for line in open(infile):
        templine=line.strip('\n')       
        for item in dic:
            if item in templine:
                #if(counts[item]<thres):            
                templine=templine.replace(item, dic[item])
                count+=1       		
        outs.write(templine+'\n')
    print(count,' multiple words fused')        
#python rewrite3.py train.ch dev.ch test.ch cedict.nos cedict.ent
#python rewrite3.py Spoken_mult_bpe/train.ch Spoken_mult_bpe/dev.ch Spoken_mult_bpe/test.ch cedict.nos Spoken_bpe/cedict.ent 10

if __name__ == "__main__":
    import sys
    import os
    src_path=str(sys.argv[1])
    trg_path=str(sys.argv[2])
    src_lang=str(sys.argv[3])
    mult=str(sys.argv[4])
    sing=str(sys.argv[5])
    thres=int(sys.argv[6])    
    
    infile=os.path.join(src_path,'train.'+src_lang)
    outfile=os.path.join(trg_path,'train.'+src_lang)  
    
    vocab= load_dictionary(mult, sing)
    #print(vocab)
    vocab,counts =refine_revdict(infile,vocab,thres)
    #print(counts)  
    #load_counts(infile,vocab)
    rewrite(infile, outfile, vocab, counts, thres)
    
    infile=os.path.join(src_path,'dev.'+src_lang)
    outfile=os.path.join(trg_path,'dev.'+src_lang)    
    rewrite(infile, outfile, vocab, counts, thres) 
    
    infile=os.path.join(src_path,'test.'+src_lang)
    outfile=os.path.join(trg_path,'test.'+src_lang)    
    rewrite(infile, outfile, vocab, counts, thres)
        
    #rewrite(dev, dev+'.rewrite', vocab) 
    #rewrite(test, test+'.rewrite', vocab) 
    
    
          
