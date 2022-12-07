#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -$JOB_NAME
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 3GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s194368@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J.out
#BSUB -e Error%J.err

# here follow the commands you want to execute

#module load python3/3.8.2


bash bash_seq_runner.sh
source /zhome/e7/a/137819/deep_venv/bin/activate
/zhome/e7/a/137819/deep_venv/bin/python3 -u /zhome/06/a/147115/02456_project_group_72/src/chatbot.py > /zhome/e7/a/137819/02456_project_group_72/seq_Ja/logs/$JOB_NAME/output.txt