#!/usr/bin/bash

# Parse options
while getopts "e:" opt; do
  case $opt in
    e) e_value=$OPTARG ;; # training episode length, units of 100
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
  esac
done

# Check if -s and -m options were provided
if [ -z "$e_value" ] ; then
  echo "-e (episode length) option is required."
  exit 1
fi

./run8.sh -s 0 -m 0 -e "$e_value"
./run8.sh -s 0 -m 1 -e "$e_value"
./run8.sh -s 0 -m 2 -e "$e_value"
./run8.sh -s 0 -m 3 -e "$e_value"
./run8.sh -s 1 -m 0 -e "$e_value"
./run8.sh -s 1 -m 1 -e "$e_value"
./run8.sh -s 1 -m 2 -e "$e_value"
./run8.sh -s 1 -m 3 -e "$e_value"
