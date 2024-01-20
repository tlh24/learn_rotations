#!/usr/bin/bash

# this is to run 8 instances of rotate in parallel
# and thereby saturate my 2 GPUs. 

python rotate.py -o 0 -l 16 -c 0 & 
python rotate.py -o 2 -l 16 -c 0 &
python rotate.py -o 4 -l 16 -c 0 &
python rotate.py -o 6 -l 16 -c 0 &
python rotate.py -o 8 -l 16 -c 1 & 
python rotate.py -o 10 -l 16 -c 1 &
python rotate.py -o 12 -l 16 -c 1 &
python rotate.py -o 14 -l 16 -c 1 &
wait
