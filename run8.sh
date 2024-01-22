#!/usr/bin/bash

# this is to run 8 instances of rotate in parallel
# and thereby saturate my 2 GPUs. 
# Initialize variables for -s and -m options
s_value=""
m_value=""
e_value=""

# Parse options
while getopts "s:m:e:" opt; do
  case $opt in
    s) s_value=$OPTARG ;; # scale singular values, 0 or 1
    m) m_value=$OPTARG ;; # mode, see source
    e) e_value=$OPTARG ;; # training episode length, units of 100
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
  esac
done

# Check if all options were provided
if [ -z "$s_value" ] || [ -z "$m_value" ] || [ -z "$e_value" ]; then
  echo "-s, -m, and -e options are required."
  exit 1
fi

python rotate.py -o 0 -l 16 -c 0 -s "$s_value" -m "$m_value" -e "$e_value" & 
python rotate.py -o 2 -l 16 -c 0 -s "$s_value" -m "$m_value" -e "$e_value" &
python rotate.py -o 4 -l 16 -c 0 -s "$s_value" -m "$m_value" -e "$e_value" &
python rotate.py -o 6 -l 16 -c 0 -s "$s_value" -m "$m_value" -e "$e_value" &
python rotate.py -o 8 -l 16 -c 1 -s "$s_value" -m "$m_value" -e "$e_value" &
python rotate.py -o 10 -l 16 -c 1 -s "$s_value" -m "$m_value" -e "$e_value" &
python rotate.py -o 12 -l 16 -c 1 -s "$s_value" -m "$m_value" -e "$e_value" &
python rotate.py -o 14 -l 16 -c 1 -s "$s_value" -m "$m_value" -e "$e_value" &
wait
