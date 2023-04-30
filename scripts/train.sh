#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PYTHONPATH=$SCRIPT_DIR/../src
source /netscratch/saifullah/envs/xai_torch/bin/activate
python $SCRIPT_DIR/../train.py --save_folder ./results_2013 --data_path /ds/images/dibco/DIBCOSETS --dataset 2013