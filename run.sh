#!/bin/bash
python inference.py \
  --config=configs/inceptionv3.attention.policy.per_image_norm.1024.yml \
  --num_tta=8 \
  --output=inferences/inceptionv3.0.test_val.csv \
  --checkpoint=swa.10.027.pth \
  --split=test_val
  
python inference.py \
  --config=configs/se_resnext50.attention.policy.per_image_norm.1024.yml \
  --num_tta=8 \
  --output=inferences/se_resnext50.0.test_val.csv \
  --checkpoint=swa.10.022.pth \
  --split=test_val
  
python tools/find_data_leak.py
python make_submission.py

shutdown -h now