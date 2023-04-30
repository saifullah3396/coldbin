#!/bin/bash

export PYTHONPATH=$SCRIPT_DIR/../src
source /netscratch/saifullah/envs/xai_torch/bin/activate
python $SCRIPT_DIR/../test.py --save_folder ./results_2013 --data_path /ds/images/dibco/DIBCOSETS --dataset 2013 --load_path /results_2013/model.pt