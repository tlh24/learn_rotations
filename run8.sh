#!/usr/bin/bash

# this is to run 8 instances of rotate in parallel
# and thereby saturate my 2 GPUs. 

python rotate.py -o 0 -l 8 -c 0 & 
python rotate.py -o 1 -l 8 -c 0 &
python rotate.py -o 2 -l 8 -c 0 &
python rotate.py -o 3 -l 8 -c 0 &
python rotate.py -o 4 -l 8 -c 1 & 
python rotate.py -o 5 -l 8 -c 1 &
python rotate.py -o 6 -l 8 -c 1 &
python rotate.py -o 7 -l 8 -c 1 &
wait
