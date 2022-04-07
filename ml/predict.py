"""
Predict appliance using novel data from my home.

Copyright (c) 2022 Lindo St. Angel
"""

import os
import argparse
import socket

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from logger import log
from common import WindowGenerator, params_appliance
from nilm_metric import get_Epd

PANEL_LOCATION = 'garage'
WINDOW_LENGTH = 599
AGGREGATE_MEAN = 522
AGGREGATE_STD = 814

default_appliances = ['kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher']
test_file_path = '/home/lindo/Develop/nilm/ml/dataset_management/my-house/'
test_file_name = 'garage.csv'

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict appliance\
                                     given a trained neural network\
                                     for energy disaggregation -\
                                     network input = mains window;\
                                     network target = the states of\
                                     the target appliance.')
    parser.add_argument('--appliances',
                        type=str,
                        nargs='+',
                        default=default_appliances,
                        help='Name(s) of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='./dataset_management/refit',
                        help='this is the directory to the test data')
    parser.add_argument('--trained_model_dir',
                        type=str,
                        default='./models',
                        help='this is the directory to the trained models')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        default='checkpoints',
                        help='directory name of model checkpoint')
    parser.add_argument('--save_results_dir',
                        type=str,
                        default='./results',
                        help='this is the directory to save the predictions')
    parser.add_argument('--show_plot', action='store_true',
                        help='If set, show plot the predicted appliance and mains power.')
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='To use part of the dataset for testing.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1000,
                        help='Sets mini-batch size.')
    parser.set_defaults(show_plot=False)
    return parser.parse_args()

def load_dataset(file_name, crop=None) -> np.array:
    """Load input dataset file and return as np array.."""
    df = pd.read_csv(file_name, header=None, nrows=crop)

    return np.array(df, dtype=np.float32)

if __name__ == '__main__':
    log(f'Machine name: {socket.gethostname()}')
    args = get_arguments()
    log('Arguments: ')
    log(args)

    log(f'Target appliance(s): {args.appliances}')

    # offset parameter from window length
    offset = int(0.5 * (WINDOW_LENGTH - 1.0))

    test_set_x = load_dataset(os.path.join(
        test_file_path, test_file_name), args.crop)
    ts_size = test_set_x.size
    log(f'There are {ts_size/10**6:.3f}M test samples.')

    test_provider = WindowGenerator(
        dataset=(test_set_x.flatten(), None),
        offset=offset,
        train=False,
        shuffle=False,
        batch_size=args.batch_size)
    
    def prediction(appliance) -> np.array:
        """Make appliance prediction and return post-processed result."""
        log(f'Making prediction for {appliance}.')
        model_file_path = os.path.join(
            args.trained_model_dir, appliance, args.ckpt_dir)
        log(f'Loading saved model from {model_file_path}.')
        model = tf.keras.models.load_model(model_file_path)
        model.summary()
        prediction = model.predict(
            x=test_provider,
            verbose=1,
            workers=24,
            use_multiprocessing=True)
        # De-normalize.
        mean = params_appliance[appliance]['mean']
        std = params_appliance[appliance]['std']
        log(f'appliance_mean: {str(mean)}')
        log(f'appliance_std: {str(std)}')
        prediction = prediction * std + mean
        # Zero out any negative power.
        prediction[prediction <= 0.0] = 0.0
        # Apply on-power thresholds.
        threshold = params_appliance[appliance]['on_power_threshold']
        prediction[prediction <= threshold] = 0.0
        return prediction
    predictions = {appliance : prediction(appliance) for appliance in args.appliances}

    log('aggregate_mean: ' + str(AGGREGATE_MEAN))
    log('aggregate_std: ' + str(AGGREGATE_STD))
    aggregate = test_set_x.flatten() * AGGREGATE_STD + AGGREGATE_MEAN

    # Calculate metrics.
    SAMPLE_PERIOD = 8 # Sampling period in seconds. 
    # get_Epd returns a relative metric between two powers, so zero out one.
    target = np.zeros_like(aggregate)
    aggregate_epd = get_Epd(target, aggregate, SAMPLE_PERIOD)
    log(f'Aggregate energy: {aggregate_epd/1000:.3f} kWh')
    for appliance in args.appliances:
        epd = get_Epd(target, predictions[appliance], SAMPLE_PERIOD)
        log(f'{appliance} energy: {epd/1000:.3f} kWh')

    save_path = os.path.join(args.save_results_dir, PANEL_LOCATION)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Find max value in predictions for setting plot limits.
    max_pred = np.ceil(np.max(list(predictions.values())))

    # Save and perhaps show powers in a single row of subplots.
    nrows = len(args.appliances) + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=1, constrained_layout=True)
    ax[0].set_ylabel('Watts')
    ax[0].set_title('aggregate')
    ax[0].plot(aggregate[offset:-offset], color='#7f7f7f', linewidth=1.8)
    row = 1
    for appliance in args.appliances:
        ax[row].set_ylabel('Watts')
        ax[row].set_title(appliance)
        ax[row].set_ylim(0, max_pred)
        ax[row].plot(predictions[appliance], color='#1f77b4', linewidth=1.5)
        row+=1
    fig.suptitle('Test results on {:}'
        .format(test_file_name), fontsize=16, fontweight='bold')
    plot_savepath = os.path.join(save_path, f'{PANEL_LOCATION}.png')
    plt.savefig(fname=plot_savepath)
    if args.show_plot:
        plt.show()
    plt.close()