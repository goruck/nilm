"""
Scale datasets created by create_new_dataset.py.

Copyright (c) 2023 Lindo St. Angel
"""

import os
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(file_name, crop=None):
    """Load input dataset file."""
    df = pd.read_csv(file_name, header=0, nrows=crop)

    return df

def compute_stats(df) -> dict:
    """ Given a Series DataFrame compute its statistics. """
    return {
        'mean': df.mean(),
        'std': df.std(),
        'median': df.median(),
        'quartile1': df.quantile(q=0.25, interpolation='lower'),
        'quartile3': df.quantile(q=0.75, interpolation='lower')
    }

def get_zscore(value, values):
    """Obtain the z-score of a given value"""
    m = np.mean(values)
    s = np.std(values)
    z_score = (value - m)/s
    return np.abs(z_score)

if __name__ == '__main__':
    default_appliance = 'kettle'
    default_dataset_dir = '/home/lindo/Develop/nilm/ml/dataset_management/refit/'

    parser = argparse.ArgumentParser(
        description='scale a dataset'
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

    # Get statistics from training dataset.
    train_file_name = os.path.join(path, f'{args.appliance}_training_.csv')
    try:
        df = load(train_file_name)

        # Remove outliers.
        #df = df[df < 10 * df.iloc[:,0].std()]

        train_agg_mean = df.iloc[:,0].mean()
        train_agg_std = df.iloc[:,0].std()
        print(f'Training aggregate mean = {train_agg_mean}, std = {train_agg_std}')

        train_app_mean = df.iloc[:,1].mean()
        train_app_std = df.iloc[:,1].std()
        print(f'Training appliance mean = {train_app_mean}, std = {train_app_std}')

        train_app_min = df.iloc[:,1].min()
        train_app_max = df.iloc[:,1].max()
        print(f'Training appliance min = {train_app_min}, max = {train_app_max}')

        del df
    except Exception as e:
        sys.exit(e)

    # Standardize (or normalize) each dataset.
    for _, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        
        df = load(file_path)

        print(f'\nStatistics for {file_name}:')
        print(df.iloc[:,0].describe())
        print(df.iloc[:,1].describe())

        # Standardize aggregate dataset using its mean and training aggregate std.
        mean = df.iloc[:,0].mean()
        print(f'\nStandardizing aggregate dataset with mean = {mean} and std = {train_agg_std}.')
        df.iloc[:,0] = (df.iloc[:,0] - mean) / train_agg_std

        # Standardize appliance dataset using its mean and training appliance std.
        mean = df.iloc[:,1].mean()
        print(f'\nStandardizing appliance dataset with mean = {mean} and std = {train_app_std}.')
        df.iloc[:,1] = (df.iloc[:,1] - mean) / train_app_std

        ### Other ways of scaling the datasets are commented out below ###
        ### The current method seems to give the best results ###

        # Remove outliers.
        # compute z-scores for all values
        # THIS TAKES FOREVER - DO NOT USE
        #df['z-score'] = df[args.appliance].apply(lambda x: get_zscore(x, df[args.appliance]))
        #outliers = df[df['z-score'] > 6]
        #print(outliers)
        #exit()
        
        #print(f'\nStatistics for {file_name} without outliers:')
        #print(df.iloc[:,0].describe())
        #print(df.iloc[:,1].describe())

        # Standardize datasets with training parameters.
        #print(f'\nUsing aggregate training statistics for both datasets.')
        #df.iloc[:,0] = (df.iloc[:,0] - agg_mean) / agg_std
        #df.iloc[:,1] = (df.iloc[:,1] - agg_mean) / agg_std

        # Standardize datasets with respective training parameters.
        #print(f'\nUsing respective training statistics for datasets.')
        #df.iloc[:,0] = (df.iloc[:,0] - agg_mean) / agg_std
        #df.iloc[:,1] = (df.iloc[:,1] - app_mean) / app_std

        # Standardize aggregate dataset.
        #print('\nStandardizing aggregate dataset with training parameters.')
        #df.iloc[:,0] = (df.iloc[:,0] - agg_mean) / agg_std

        # Standardize appliance dataset.
        #print('\nStandardizing appliance dataset with its mean and active train std from NILMTK.')
        #df.iloc[:,1] = (df.iloc[:,1] - df.iloc[:,1].mean()) / 2.140622e+01
        #df.iloc[:,1] = df.iloc[:,1].clip(lower=0.0)

        # Normalize appliance dataset using training parameters.
        #print(f'\nNormalizing appliance dataset with min = {min} and max = {max}.')
        #df.iloc[:,1] = (df.iloc[:,1] - min) / (max - min)

        # Normalize datasets with average training parameters.
        #print(f'\nUsing normalization params mean: {str(545.0)}, std: {str(820.0)}')
        #df.iloc[:,0] = (df.iloc[:,0] - 545.0) / 820.0
        #df.iloc[:,1] = (df.iloc[:,1] - 545.0) / 820.0

        # Normalize appliance dataset to [0, 1].
        #min = df.iloc[:,1].min()
        #max = df.iloc[:,1].max()
        #print(f'\nNormalizing appliance dataset with min = {min} and max = {max}')
        #df.iloc[:,1] = (df.iloc[:,1] - min) / (max - min)

        print(f'\nStatistics for {file_name} after scaling:')
        print(df.iloc[:,0].describe())
        print(df.iloc[:,1].describe())

        # Show dataset histograms.
        df.iloc[:,0].hist()
        plt.title(f'Histogram for {file_name} aggregate')
        plt.show()
        df.iloc[:,1].hist()
        plt.title(f'Histogram for {file_name} {args.appliance}')
        plt.show()

        # Check for NaNs.
        print(f'\nNaNs present: {df.isnull().values.any()}')

        # Save scaled dataset and overwrite existing csv.
        print(f'\nSaving dataset to {file_path}.')
        df.to_csv(file_path, index=False)

        del df