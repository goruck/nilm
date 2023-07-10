#from nilm.Arguments import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

appliance_name = 'kettle'

#dataset = 'training'
dataset = 'test'
#dataset = 'validation'
#dataset = 'train'

DATA_DIRECTORY = '/home/lindo/Develop/nilm/ml/dataset_management/refit/'

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the CLEAN REFIT data')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    return parser.parse_args()

args = get_arguments()

for filename in os.listdir(args.data_dir + appliance_name):
        if dataset == 'train' and dataset.upper() in filename.upper() and 'TEST' in filename.upper():
            test_filename = filename
        elif dataset == 'training' and dataset.upper() in filename.upper():
            test_filename = filename
        elif dataset == 'test' and dataset.upper() in filename.upper() and 'train' not in filename.upper():
            test_filename = filename
        elif dataset == 'validation' and dataset.upper() in filename.upper():
            test_filename = filename

chunksize = 10 ** 6

for idx, chunk in enumerate(pd.read_csv(args.data_dir + appliance_name + '/' + 'kettle_training_.csv',
                                        # index_col=False,
                                        names=['aggregate', appliance_name, 'status'],
                                        # usecols=[1, 2],
                                        # iterator=True,
                                        #skiprows=15 * 10 ** 6,
                                        chunksize=chunksize,
                                        header=0
                                        )):

    # de-normalization
    #chunk['aggregate'] = chunk['aggregate'] * 822 + 522
    #chunk[appliance] = chunk[appliance] * params_appliance[args.appliance_name]['std'] \
                      #+ params_appliance[args.appliance_name]['mean']


    fig = plt.figure(num='Figure {:}'.format(idx))
    ax1 = fig.add_subplot(111)

    ax1.plot(chunk['aggregate'])
    ax1.plot(chunk[appliance_name])
    ax1.plot(chunk['status'])

    ax1.grid()
    ax1.set_title('{:}'.format(test_filename), fontsize=14, fontweight='bold')
    ax1.set_ylabel('Power normalized')
    ax1.set_xlabel('samples')
    ax1.legend(['aggregate', appliance_name, 'status'])

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    del chunk