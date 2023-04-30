#!/bin/bash

export PYTHONPATH=$SCRIPT_DIR/../src
python $SCRIPT_DIR/../test.py --save_folder ./results_2013 --data_path /path/to/dibcosets --dataset 2013 --load_path /results_2013/model.pt