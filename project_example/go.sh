#!/bin/bash -ex       

python main.py --model "configs/model/RandomForest.py" \
    	       --data "configs/data/RAVDESS.py" \
	       --features "configs/features/egemaps.py"