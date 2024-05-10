"""
Create an aggregate dataset from my house data. Optionally add ground truth.

No scaling is performed.

Copyright (c) 2023~2024 Lindo St. Angel
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import common

DATA_DIRECTORY = '/home/lindo/Develop/nilm-datasets/my-house/house'
FILE_NAME = 'samples_02_20_23.csv'
GROUND_TRUTH_FILE_NAME = 'appliance_energy_data_022023.csv'
SAVE_PATH = '/home/lindo/Develop/nilm/ml/dataset_management/my-house/house.csv'

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Create aggregate dataset from house data'
    )
    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIRECTORY,
        help='directory containing my house data'
    )
    parser.add_argument(
        '--file_name', type=str, default=FILE_NAME,
        help='file name containing the csv data'
    )
    parser.add_argument(
        '--gt_file_name', type=str, default=GROUND_TRUTH_FILE_NAME,
        help='file name containing the ground truth csv data'
    )
    parser.add_argument(
        '--save_path', type=str, default=SAVE_PATH,
        help='directory to store the dataset.'
    )
    parser.add_argument(
        '--add_gt', action='store_true',
        help='if set, add appliance ground truth'
    )
    parser.set_defaults(add_gt=False)
    return parser.parse_args()

def get_real_power(file_name, zone='US/Pacific') -> pd.DataFrame:
    """Load input dataset and return total real power with datetimes."""

    df = pd.read_csv(
        file_name,
        usecols=['DT', 'W1', 'W2'],
        na_filter=False,
        parse_dates=['DT'],
        date_format='ISO8601'
    )

    # Get datetimes and localize.
    df_dt = df['DT'].dt.tz_convert(tz=zone)
    # Compute total real power.
    df_rp = df['W1'] + df['W2']

    return pd.concat([df_dt, df_rp], axis=1, keys=['date_time', 'real_power'])

def get_ground_truth(file_name, crop=None, zone='US/Pacific') -> pd.DataFrame:
    """Load ground truth dataset and return appliance active power with datetimes."""
    dataframe = pd.read_csv(
        file_name,
        nrows=crop,
        usecols=['date', 'appliance', 'apower'],
        na_filter=False,
        parse_dates=['date'],
        date_format='ISO8601'
    )

    # Get datetimes and localize.
    df_dt = dataframe['date'].dt.tz_convert(tz=zone)

    # Make columns of appliance data into rows.
    # Note apower is active power which is same as real power.
    df_ap = pd.pivot(dataframe, columns='appliance', values='apower')
    # Convert NaN's into zero's.
    df_ap = df_ap.fillna(0)

    # Concat datetimes and appliance active power into single dataframe.
    df_ap = pd.concat([df_dt, df_ap], axis=1)

    # Make column names consistent with convention.
    df_ap.rename(columns={'date': 'date_time', 'washing machine': 'washingmachine'}, inplace=True)

    # Downsample to match global sample rate.
    # Although the appliance ground truth power was sampled at 1 data point every 8 seconds,
    # each appliance was read consecutively by the logging program with about 1 second
    # between reads. Therefore the dataset has groups of appliance readings clustered
    # around 8 second intervals that this downsampling corrects for.
    resample_period = f'{common.SAMPLE_PERIOD}s'
    return df_ap.resample(resample_period, on='date_time').max()

def main():
    args = get_arguments()

    path = os.path.join(args.data_dir, args.file_name)

    print(f'Input file path is {path}')

    save_path = args.save_path

    print('Creating dataset.')

    # Get real power.
    df = get_real_power(path)

    print(f'\nRaw dataset statistics: \n{df["real_power"].describe()}')

    # Show dataset histograms.
    df['real_power'].hist()
    plt.title(f'Histogram for {args.file_name} aggregate')
    plt.show()

    print(df)

    if args.add_gt:
        print('Adding ground truth.')
        path = os.path.join(args.data_dir, args.gt_file_name)
        gt = get_ground_truth(file_name=path)
        df = pd.merge_asof(df, gt, on='date_time')
        print(df)

    # Save.
    df.to_csv(save_path, header=True, index=False)

    print(f'Created dataset and saved to {save_path}')
    print(f'Size of dataset is {df.shape[0]} rows.')

if __name__ == '__main__':
    main()
