#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PYTHONPATH=$SCRIPT_DIR/../src
python $SCRIPT_DIR/../train.py --save_folder ./results_2013 --data_path /path/to/dibcosets --dataset 2013