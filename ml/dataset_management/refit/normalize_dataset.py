"""Scale datasets created by create_new_dataset.py and add on-off status.

Copyright (c) 2023 Lindo St. Angel
"""

import os
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../../ml')
import common

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

    appliance = args.appliance

    print(f'Target appliance: {appliance}')

    path = os.path.join(args.datadir, appliance)

    # Get statistics from training dataset.
    train_file_name = os.path.join(path, f'{appliance}_training_.csv')
    try:
        df = load(train_file_name)
        aggregate_power = df.loc[:, 'aggregate']
        appliance_power = df.loc[:, appliance]

        train_agg_mean = aggregate_power.mean()
        train_agg_std = aggregate_power.std()
        print(f'Training aggregate mean = {train_agg_mean}, std = {train_agg_std}')

        train_app_mean = appliance_power.mean()
        train_app_std = appliance_power.std()
        print(f'Training appliance mean = {train_app_mean}, std = {train_app_std}')

        train_app_min = appliance_power.min()
        train_app_max = appliance_power.max()
        print(f'Training appliance min = {train_app_min}, max = {train_app_max}')

        del df
    except Exception as e:
        sys.exit(e)

    max_on_power = common.params_appliance[appliance]['max_on_power']

    # Standardize (or normalize) each dataset and add status.
    for _, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        
        df = load(file_path)

        print(f'\n*** Working on {file_name} ***')
        print('Raw dataset statistics:')
        print(df.loc[:, 'aggregate'].describe())
        print(df.loc[:, appliance].describe())

        # Limit appliance power to [0, max_on_power].
        print(f'Limiting appliance power to [0, {max_on_power}]')
        df.loc[:, appliance] = df.loc[:, appliance].clip(0, max_on_power)

        # Get appliance status and add to end of dataframe.
        print('Computing on-off status.')
        status = common.compute_status(df.loc[:, appliance].to_numpy(), appliance)
        df.insert(2, 'status', status)
        num_on = len(df[df["status"]==1])
        num_off = len(df[df["status"]==0])
        print(f'Number of samples with on status: {num_on}')
        print(f'Number of samples with off status: {num_off}')
        assert num_on + num_off == df.iloc[:, 2].size

        # Standardize aggregate dataset.
        agg_mean = common.ALT_AGGREGATE_MEAN if common.USE_ALT_STANDARDIZATION else train_agg_mean
        agg_std = common.ALT_AGGREGATE_STD if common.USE_ALT_STANDARDIZATION else train_agg_std
        print(f'Standardizing aggregate dataset with mean = {agg_mean} and std = {agg_std}.')
        df.loc[:, 'aggregate'] = (df.loc[:, 'aggregate'] - agg_mean) / agg_std

        # Scale appliance dataset.
        if common.USE_APPLIANCE_NORMALIZATION:
            # Normalize appliance dataset to [0, max_on_power].
            min = 0
            max = max_on_power
            print(f'Normalizing appliance dataset with min = {min} and max = {max}.')
            df.loc[:, appliance] = (df.loc[:, appliance] - min) / (max - min)
        else:
            # Standardize appliance dataset.
            alt_app_mean = common.params_appliance[appliance]['alt_app_mean']
            alt_app_std = common.params_appliance[appliance]['alt_app_std']
            app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
            app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std
            print('Using alt standardization.' if common.USE_ALT_STANDARDIZATION 
                  else 'Using default standardization.')
            print(f'Standardizing appliance dataset with mean = {app_mean} and std = {app_std}.')
            df.loc[:, appliance] = (df.loc[:, appliance] - app_mean) / app_std

        ### Other ways of scaling the datasets are commented out below ###
        ### The current method seems to give the best results ###

        # Remove outliers.
        # compute z-scores for all values
        # THIS TAKES FOREVER - DO NOT USE
        #df['z-score'] = df[appliance].apply(lambda x: get_zscore(x, df[appliance]))
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
        #print(f'Normalizing appliance dataset with min = {min} and max = {max}')
        #df.iloc[:, 1] = (df.iloc[:, 1] - min) / (max - min)

        print(f'Statistics for {file_name} after scaling:')
        print(df.loc[:, 'aggregate'].describe())
        print(df.loc[:, appliance].describe())

        # Show dataset histograms.
        df.loc[:, 'aggregate'].hist()
        plt.title(f'Histogram for {file_name} aggregate')
        plt.show()
        df.loc[:, appliance].hist()
        plt.title(f'Histogram for {file_name} {appliance}')
        plt.show()

        # Check for NaNs.
        print(f'NaNs present: {df.isnull().values.any()}')

        # Save scaled dataset and overwrite existing csv.
        print(f'*** Saving dataset to {file_path}. ***')
        df.to_csv(file_path, index=False)

        del df