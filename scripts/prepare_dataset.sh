#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
python $SCRIPT_DIR/prepare_dibco_dataset.py --data_path /path/to/dibcosets/ --split_size 256 --testing_dataset 2013 --validation_dataset 2019