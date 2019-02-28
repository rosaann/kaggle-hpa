#!/bin/bash
python tools/find_duplicate_images.py
python tools/stratified_split.py
python tools/stratified_split.py --use_external=0
python train.py --config=configs/inceptionv3.attention.policy.per_image_norm.1024.yml