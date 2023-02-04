"""
Create an aggregate dataset from my house data.

No scaling is performed.

Copyright (c) 2023 Lindo St. Angel
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIRECTORY = '/home/lindo/Develop/nilm-datasets/my-house/house'
FILE_NAME = 'samples_10_31_22.csv'
SAVE_PATH = '/home/lindo/Develop/nilm/ml/dataset_management/my-house/house.csv'

def get_arguments():
    parser = argparse.ArgumentParser(description='Create aggregate dataset from house data')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing my house data')
    parser.add_argument('--file_name', type=str, default=FILE_NAME,
                          help='The file name containing the csv data')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the dataset.')
    return parser.parse_args()

def get_real_power(file_name, zone='US/Pacific') -> pd.DataFrame:
    """Load input dataset and return total real power with datetimes."""

    df = pd.read_csv(file_name,
        usecols=['DT', 'W1', 'W2'],
        na_filter=False,
        parse_dates=['DT'],
        infer_datetime_format=True)

    # Get datetimes and localize.
    df_dt = df['DT'].dt.tz_localize(tz=zone)
    # Compute total real power.
    df_rp = df['W1'] + df['W2']

    return pd.concat([df_dt, df_rp], axis=1, keys=['date_time', 'real_power'])

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

    # Save.
    df.to_csv(save_path, header=True, index=False)

    print(f'Created dataset and saved to {save_path}')
    print(f'Size of dataset is {df.shape[0]} rows.')
    
if __name__ == '__main__':
    main()