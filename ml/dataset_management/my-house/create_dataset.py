from numpy import append
import pandas as pd
import time
import os
import re
import argparse

DATA_DIRECTORY = '/home/lindo/Develop/nilm-datasets/my-house/garage'
FILE_NAME = 'samples_4_4_22.csv'
SAVE_PATH = '/home/lindo/Develop/nilm/ml/dataset_management/my-house/garage.csv'
AGG_MEAN = 522#242.24
AGG_STD = 814#15.26

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing my house data')
    parser.add_argument('--file_name', type=str, default=FILE_NAME,
                          help='The file name containing the csv data')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    return parser.parse_args()

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
        'channels': [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
        'test_house': 2,
        'validation_house': 5,
        'test_on_train_house': 5,
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [4, 10, 12, 17, 19],
        'channels': [8, 8, 3, 7, 4],
        'test_house': 4,
        'validation_house': 17,
        'test_on_train_house': 10,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [2, 5, 9, 12, 15],
        'channels': [1, 1, 1,  1, 1],
        'test_house': 15,
        'validation_house': 12,
        'test_on_train_house': 5,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [5, 7, 9, 13, 16, 18, 20],
        'channels': [4, 6, 4, 4, 6, 6, 5],
        'test_house': 20,
        'validation_house': 18,
        'test_on_train_house': 13,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [2, 5, 7, 8, 9, 15, 16, 17, 18],
        'channels': [2, 3, 5, 4, 3, 3, 5, 4, 5],
        'test_house': 8,
        'validation_house': 18,
        'test_on_train_house': 5,
    }
}

def get_app_pwr(file_name) -> pd.DataFrame:
    """Load input dataset and return total apparent power."""

    df = pd.read_csv(file_name,
        header=0,
        names=['VA1', 'VA2'],
        usecols=[5, 8],
        na_filter=False,
        parse_dates=True,
        infer_datetime_format=True)

    return df.iloc[:,0] + df.iloc[:,1]

def main():
    args = get_arguments()
    
    path = os.path.join(args.data_dir, args.file_name)

    print(f'Input file path is {path}')

    save_path = args.save_path

    aggregate_mean = args.aggregate_mean
    aggregate_std = args.aggregate_std

    print('Creating dataset.')

    # Get apparent power. 
    app_pwr = get_app_pwr(path)

    # Normalize.
    app_pwr = (app_pwr - aggregate_mean) / aggregate_std

    # Save.
    app_pwr.to_csv(save_path, header=False, index=False)

    print(f'Created dataset and saved to {save_path}')
    print(f'Size of dataset is {app_pwr.shape[0]} rows.')

    del app_pwr
    
    """
    print("Size of training set is {:.3f} M rows.".format(total_length / 10 ** 6))
    print("\nNormalization parameters: ")
    print("Mean and standard deviation values USED for AGGREGATE are:")
    print("    Mean = {:d}, STD = {:d}".format(aggregate_mean, aggregate_std))
    print('Mean and standard deviation values USED for ' + appliance_name + ' are:')
    print("    Mean = {:d}, STD = {:d}"
          .format(params_appliance[appliance_name]['mean'], params_appliance[appliance_name]['std']))
    print("\nTraining, validation and test sets are  in: " + save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    """
    
if __name__ == '__main__':
    main()