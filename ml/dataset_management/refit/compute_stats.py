"""Compute statistics on datasets"""

import pandas as pd
import os
import argparse

import numpy as np

def load(file_name, crop=None) -> np.array:
    """Load input dataset file and return as np array.."""
    df = pd.read_csv(file_name, header=0, nrows=crop)

    return np.array(df, dtype=np.float32)

if __name__ == '__main__':
    default_appliance = 'fridge'
    default_dataset_dir = '/home/lindo/Develop/nilm/ml/dataset_management/refit/'

    parser = argparse.ArgumentParser(
        description='Compute statistics on dataset'
    )
    parser.add_argument('--appliance',
                        type=str,
                        default=default_appliance,
                        help='name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default=default_dataset_dir,
                        help='directory location dataset')
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='use part of the dataset for calculations')

    args = parser.parse_args()

    print(f'Target appliance: {args.appliance}')

    path = os.path.join(args.datadir, args.appliance)

    for _, file_name in enumerate(os.listdir(path)):
        dataset = load(os.path.join(path, file_name), crop=args.crop)

        print(f'Computed statistics for {file_name}:')
        computed_agg_mean = np.mean(dataset[0])
        computed_agg_std = np.std(dataset[0])
        print(f'Computed agg mean: {computed_agg_mean} computed agg std: {computed_agg_std}')
        computed_app_mean = np.mean(dataset[1])
        computed_app_std = np.std(dataset[1])
        print(f'Computed app mean: {computed_app_mean} computed app std: {computed_app_std}')

        del dataset