from __future__ import print_function, division
import os
from string import Template
template = Template("""#!/bin/bash
#$$ -M xzhong3@nd.edu
#$$ -m ae
#$$ -r n
#$$ -q $queue
#$$ -l gpu_card=$gpu_card
#$$ -N $name         # Specify job name

conda activate pytorch
module load cuda/10.0
python -m nmt --proto $name
""")

def write_script(name,queue,gpu_card):

    job_script = template.substitute(dict(queue=queue, name=name,gpu_card=gpu_card))
    job_script_path = str(name)+'.sh'
    open(job_script_path, 'w').close()
    with open(job_script_path, 'w') as f:
        f.write(job_script)

if __name__ == "__main__":
    import sys
    write_script(str(sys.argv[1]),str(sys.argv[2]),int(sys.argv[3]))
