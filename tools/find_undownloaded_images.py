#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:27:33 2019

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

    external_filenames = list(glob.glob(os.path.join(external_dir, '*_green.png')))
   # print(external_filenames)
    df_external = pd.read_csv(os.path.join(args.data_dir, 'external.csv'))
    records = []
    for _, row in tqdm.tqdm(df_external.iterrows()):
        id_t = row['Id']
        this_dir = os.path.join(external_dir, id_t)
        this_dir = this_dir + '_green.png'
        #print('this_dir ', this_dir)
        #break
        if this_dir in external_filenames:
            records.append((row['Id'], row['Target']))
            
    df = pd.DataFrame.from_records(records, columns=['Id', 'Target'])
    output_filename = os.path.join(args.data_dir, 'external_z.csv')
    df.to_csv(output_filename, index=False)
 
if __name__ == '__main__':
    main()