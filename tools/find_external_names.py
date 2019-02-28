#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:19:16 2019

@author: zl
"""

import os
import argparse
import glob
import shutil
from collections import defaultdict

import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import imagehash

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='data', type=str)
    return parser.parse_args()
def main():
    args = parse_args()
    raw_images_dir = os.path.join(args.data_dir, 'raw')
    external_dir = os.path.join(raw_images_dir, 'external')

    external_filenames = list(glob.glob(os.path.join(external_dir, '*.png')))
   # print(external_filenames)
    records = []
    for file_name in external_filenames:
        
        records.append((file_name.split('/')[-1]))
            
    df = pd.DataFrame.from_records(records, columns=['Id'])
    output_filename = os.path.join(args.data_dir, 'external_name.csv')
    df.to_csv(output_filename, index=False)
 
if __name__ == '__main__':
    main()