"""
Create new train, test and validation datasets from REFIT data.

No normalization is performed, post-process with normalize_dataset.py.

Copyright (c) 2023 Lindo St. Angel
"""

import time
import os
import re
import argparse

import pandas as pd
#import matplotlib.pyplot as plt

DATA_DIRECTORY = '/home/lindo/Develop/nilm-datasets/REFIT/CLEAN_REFIT_081116/'
SAVE_DIRECTORY = '/home/lindo/Develop/nilm/ml/dataset_management/refit'
def get_arguments():
    parser = argparse.ArgumentParser(description='create new datasets for training')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the CLEAN REFIT data')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--save_path', type=str, default=SAVE_DIRECTORY,
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

def load(path, building, appliance, channel):
    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             usecols=[2, channel+2],
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )

    return single_csv

def compute_stats(df) -> dict:
    """ Given a Series DataFrame compute its statistics. """
    return {
        'mean': df.mean(),
        'std': df.std(),
        'median': df.median(),
        'quartile1': df.quantile(q=0.25, interpolation='lower'),
        'quartile3': df.quantile(q=0.75, interpolation='lower')
    }

def main():
    start_time = time.time()
    
    args = get_arguments()
    
    appliance_name = args.appliance_name
    print(appliance_name)
    
    path = args.data_dir
    save_path = os.path.join(args.save_path, args.appliance_name)
    if not os.path.exists(save_path): os.makedirs(save_path)
    print(f'data path: {path}')
    print(f'save path: {save_path}')
    
    total_length = 0
    print("Creating datasets...")
    # Looking for proper files
    for _, filename in enumerate(os.listdir(path)):
        if filename == 'CLEAN_House' + str(params_appliance[appliance_name]['test_house']) + '.csv':
            print('File: ' + filename + ' test set')
            # Loading
            test = load(path,
                 params_appliance[appliance_name]['test_house'],
                 appliance_name,
                 params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses']
                        .index(params_appliance[appliance_name]['test_house'])]
                 )

            print(test.iloc[:, 0].describe())
            #test.iloc[:, 0].hist()
            #plt.show()
            print(test.iloc[:, 1].describe())
            #test.iloc[:, 1].hist()
            #plt.show()

            agg_stats = compute_stats(test.iloc[:, 0])
            print(f'aggregate - mean: {agg_stats["mean"]}, std: {agg_stats["std"]}')
            print(f'aggregate - median: {agg_stats["median"]}, quartile1: {agg_stats["quartile1"]}, quartile3: {agg_stats["quartile3"]}')
            app_stats = compute_stats(test.iloc[:, 1])
            print(f'{appliance_name} - mean: {app_stats["mean"]}, std: {app_stats["std"]}')
            print(f'{appliance_name} - median: {app_stats["median"]}, quartile1: {app_stats["quartile1"]}, quartile3: {app_stats["quartile3"]}')
    
            # Save
            fname = os.path.join(save_path, f'{appliance_name}_test_H{params_appliance[appliance_name]["test_house"]}.csv')
            test.to_csv(fname, index=False)
    
            print("Size of test set is {:.3f} M rows (House {:d})."
                  .format(test.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_house']))
            del test
    
        elif filename == 'CLEAN_House' + str(params_appliance[appliance_name]['validation_house']) + '.csv':
            print('File: ' + filename + ' validation set')
            # Loading
            val = load(path,
                 params_appliance[appliance_name]['validation_house'],
                 appliance_name,
                 params_appliance[appliance_name]['channels']
                 [params_appliance[appliance_name]['houses']
                        .index(params_appliance[appliance_name]['validation_house'])]
                 )
            
            print(val.iloc[:, 0].describe())
            #val.iloc[:, 0].hist()
            #plt.show()
            print(val.iloc[:, 1].describe())
            #val.iloc[:, 1].hist()
            #plt.show()

            agg_stats = compute_stats(val.iloc[:, 0])
            print(f'aggregate - mean: {agg_stats["mean"]}, std: {agg_stats["std"]}')
            print(f'aggregate - median: {agg_stats["median"]}, quartile1: {agg_stats["quartile1"]}, quartile3: {agg_stats["quartile3"]}')
            app_stats = compute_stats(val.iloc[:, 1])
            print(f'{appliance_name} - mean: {app_stats["mean"]}, std: {app_stats["std"]}')
            print(f'{appliance_name} - median: {app_stats["median"]}, quartile1: {app_stats["quartile1"]}, quartile3: {app_stats["quartile3"]}')
    
            # Save
            fname = os.path.join(save_path, f'{appliance_name}_validation_H{params_appliance[appliance_name]["validation_house"]}.csv')
            val.to_csv(fname, index=False)
    
            print("Size of validation set is {:.3f} M rows (House {:d})."
                  .format(val.shape[0] / 10 ** 6, params_appliance[appliance_name]['validation_house']))
            del val
    
        elif int(re.search(r'\d+', filename).group()) in params_appliance[appliance_name]['houses']:
            print('File: ' + filename)
            print('    House: ' + re.search(r'\d+', filename).group())
    
            # Loading
            try:
                csv = load(path,
                           int(re.search(r'\d+', filename).group()),
                           appliance_name,
                           params_appliance[appliance_name]['channels']
                           [params_appliance[appliance_name]['houses']
                                  .index(int(re.search(r'\d+', filename).group()))]
                           )

                print(csv.iloc[:, 0].describe())
                #csv.iloc[:, 0].hist()
                #plt.show()
                print(csv.iloc[:, 1].describe())
                #csv.iloc[:, 1].hist()
                #plt.show()

                agg_stats = compute_stats(csv.iloc[:, 0])
                print(f'aggregate - mean: {agg_stats["mean"]}, std: {agg_stats["std"]}')
                print(f'aggregate - median: {agg_stats["median"]}, quartile1: {agg_stats["quartile1"]}, quartile3: {agg_stats["quartile3"]}')
                app_stats = compute_stats(csv.iloc[:, 1])
                print(f'{appliance_name} - mean: {app_stats["mean"]}, std: {app_stats["std"]}')
                print(f'{appliance_name} - median: {app_stats["median"]}, quartile1: {app_stats["quartile1"]}, quartile3: {app_stats["quartile3"]}')
    
                rows, _ = csv.shape
                total_length += rows
    
                if filename == 'CLEAN_House' + str(params_appliance[appliance_name]['test_on_train_house']) + '.csv':
                    fname = os.path.join(
                        save_path,
                        f'{appliance_name}_test_on_train_H{params_appliance[appliance_name]["test_on_train_house"]}.csv')
                    csv.to_csv(fname, index=False)
                    print("Size of test on train set is {:.3f} M rows (House {:d})."
                          .format(csv.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_on_train_house']))
    
                # saving the whole merged file
                fname = os.path.join(save_path, f'{appliance_name}_training_.csv')
                # Append df to csv if it exists with header only the first time.
                csv.to_csv(fname, mode = 'a', index = False, header = not os.path.isfile(fname))
    
                del csv
    
            except:
                pass
    
    print("Size of training set is {:.3f} M rows.".format(total_length / 10 ** 6))
    print("\nTraining, validation and test sets are  in: " + save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    
if __name__ == '__main__':
    main()