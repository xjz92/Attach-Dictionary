def load_dictionary(ent_path,def_path):
    dicti={}
    for ents, defs in zip(open(ent_path),open(def_path)):
        entry=ents.strip('\n')
        defini=defs.strip('\n')
        #print(len(defini.split()))
        if(entry!=defini):
            dicti[entry]=defini
    sorted_items= sorted(dicti.items(), key=lambda item: len(item[1]), reverse=True)
    out=dict(sorted_items)
    #print(len(out))        
    return out                    
        
def rewrite(infile, outfile, dic): 
    outs=open(outfile,'w')
    count=0
    for line in open(infile):
        templine=line.strip('\n')       
        for item in dic:
            if item in templine:            
                templine=templine.replace(item, dic[item])
                count+=1       		
        outs.write(templine+'\n')
    print(count,' multiple words fused')        

if __name__ == "__main__":
    import sys 
    infile=str(sys.argv[1])
    outfile=str(sys.argv[2])
    mult=str(sys.argv[3])
    sing=str(sys.argv[4])
    vocab= load_dictionary(mult, sing)
    rewrite(infile, outfile, vocab) 
    
    
          
