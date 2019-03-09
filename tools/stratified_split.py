from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import math
from itertools import combinations
import random

import tqdm
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def merge_labels(a, b):
    a_ids = a[0]
    b_ids = b[0]
    if set(a_ids) <= set(b_ids):
        return b[1]
    elif set(b_ids) <= set(a_ids):
        return a[1]

    max_len_a = max([len(v) for v in a[0]])
    max_len_b = max([len(v) for v in b[0]])
    if max_len_a > max_len_b:
        return a[1]
    elif max_len_a < max_len_b:
        return b[1]

    return a[1]


def make_examples_dict(df, use_external):
    dup_examples_dict = {}
    single_examples_dict = {}
    for _, row in tqdm.tqdm(df.iterrows()):
        ids = tuple(sorted(row['Ids'].split(' ')))

        if not use_external:
            train_id_str_len = len('050a106a-bbc1-11e8-b2bb-ac1f6b6435d0')
            ids = tuple([v for v in ids if len(v) == train_id_str_len])

        if len(ids) == 0:
            continue
            
        labels = [int(l) for l in row['Target'].split(' ')]
        if len(ids) > 1:
            dup_examples_dict[ids] = set(labels)
        else:
            single_examples_dict[ids] = set(labels)
    return dup_examples_dict, single_examples_dict


def merge_duplicates(dup_examples_dict):
    while True:
        dup_examples = list(dup_examples_dict.items())
        len_before = len(dup_examples)

        dup_examples_dict = {}
        for a, b in tqdm.tqdm(combinations(dup_examples, 2)):
            intersection = set(a[0]) & set(b[0])
           # print('inter ', intersection)
            if len(intersection) > 0:
                union = set(a[0]) | set(b[0])
               # print('union ', union)
                mer = merge_labels(a, b)
           #     print('a ', a, ' b ', b)

           #     print('mer ', mer)
                dup_examples_dict[tuple(sorted(list(union)))] = mer
           #     print('dup_examples_dict ', dup_examples_dict)

        for example in tqdm.tqdm(dup_examples):
            is_pass = False
            for ids, target in dup_examples_dict.items():
                if set(example[0]) <= set(ids):
                    is_pass = True
                    break
            if is_pass == False:
                dup_examples_dict[example[0]] = example[1]

        len_after = len(dup_examples_dict)
        if len_before == len_after:
            break
    return dup_examples_dict


def merge_dup_and_single(dup_examples_dict, single_examples_dict):
    all_examples_dict = {}
    for key, value in tqdm.tqdm(single_examples_dict.items()):
        is_dup = False
        for dup_key in dup_examples_dict.keys():
            if set(key) <= set(dup_key):
                is_dup = True
                break
        if is_dup == False:
            all_examples_dict[key] = value

    all_examples_dict.update(dup_examples_dict)
    return all_examples_dict


def split_stratified(all_examples_dict):
    examples = []
    y_list = []
    
    for key, labels in all_examples_dict.items():
        labels = list(labels)
        np_labels = np.zeros((28,), dtype=int)
        np_labels[np.array(labels)] = 1
       # print('np_labels ', np_labels)
       # print('key ', key)
        examples.append((key, labels))
        y_list.append(np_labels)

    X = np.arange(len(y_list))
    y = np.array(y_list)

    # test_val
    mskf = MultilabelStratifiedKFold(n_splits=11, random_state=1234)
    folds = []
  #  print('whole len ', len(X))
    for train_index, test_index in mskf.split(X, y):
        folds.append(test_index)
       # print('train_index ', train_index)
       # print('train_len ', len(train_index))
        
      #  print('test_index ', test_index)
      #  print('test_index ', len(test_index))

    for a, b in combinations(folds, 2):
        assert len(set(a) & set(b)) == 0
   # print('folds ', folds)
    return examples, folds

#把数据列表，分成11个不重复的folds
#再分5次，每一次把11个folds按一定配比组合成val的，train的，test_val的,每一次的结果，存一个csv
def save(examples, folds, num_fold, data_dir, use_external):
    for fold_idx in range(num_fold):
        print('fold_idx ', fold_idx)
        records = []
        for i, indices in enumerate(folds):
            print('i ', i)
            if i == (fold_idx * 2) or i == (fold_idx * 2) + 1:
                print(' ** val')
                for j in indices:
                    for id_str in examples[j][0]:
                        d = (id_str, ' '.join([str(v) for v in examples[j][1]]), 'val')
                       # print('d ', d)
                        records.append(d)
            elif i == 10:
                print(' *** test_val')
                for j in indices:
                    for id_str in examples[j][0]:
                        records.append((id_str, ' '.join([str(v) for v in examples[j][1]]), 'test_val'))
            else:
                print(' **** train')
                for j in indices:
                    for id_str in examples[j][0]:
                        dd = (id_str, ' '.join([str(v) for v in examples[j][1]]), 'train')
                      #  print('dd ', dd)
                        records.append(dd)
        df = pd.DataFrame.from_records(records, columns=['Id', 'Target', 'Split'])
        if use_external:
            output_filename = os.path.join(data_dir, 'split.stratified.{}.csv'.format(fold_idx))
        else:
            output_filename = os.path.join(data_dir, 'split.stratified.small.{}.csv'.format(fold_idx))
        df.to_csv(output_filename, index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='data', type=str)
    parser.add_argument('--use_external', dest='use_external',
                        help='1: with external, 0: without external',
                        default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    num_fold = 5
    data_dir = args.data_dir
    use_external = args.use_external == 1

    # merge phash and ahash
    df_phash = pd.read_csv(os.path.join(data_dir, 'duplicates.phash.csv'))
    df_ahash = pd.read_csv(os.path.join(data_dir, 'duplicates.ahash.csv'))

    dup_examples_dict, single_examples_dict = make_examples_dict(df_phash, use_external)
    dup_examples_dict_a, single_examples_dict_a = make_examples_dict(df_ahash, use_external)
    dup_examples_dict.update(dup_examples_dict_a)
    single_examples_dict.update(single_examples_dict_a)

   # print(dup_examples_dict)
    print('len(dup_examples):', len(dup_examples_dict))
    print('len(single_examples):', len(single_examples_dict))

    dup_examples_dict = merge_duplicates(dup_examples_dict)
    all_examples_dict = merge_dup_and_single(dup_examples_dict, single_examples_dict)

    all_examples_dict = merge_dup_and_single(dup_examples_dict, single_examples_dict)
    print('len(all_examples_dict):', len(all_examples_dict))

    examples, folds = split_stratified(all_examples_dict)
    save(examples, folds, num_fold, data_dir, use_external)


if __name__ == '__main__':
  main()

