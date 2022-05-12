"""
Predict appliance type and power using novel data from my home.

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

WINDOW_LENGTH = 599
AGGREGATE_MEAN = 522
AGGREGATE_STD = 814
SAMPLE_PERIOD = 8 # Mains sample period in seconds.

if __name__ == '__main__':
    default_appliances = [
        'kettle', 'fridge', 'microwave', 'washingmachine', 'dishwasher'
    ]
    default_dataset_dir = '/home/lindo/Develop/nilm/ml/dataset_management/my-house'
    default_panel_location = 'garage'
    default_model_dir = '/home/lindo/Develop/nilm/ml/models'
    default_ckpt_dir = 'checkpoints'
    default_results_dir = '/home/lindo/Develop/nilm/ml/results'
    default_rt_preds_dataset_dir = '/home/lindo/Develop/nilm-datasets/my-house/garage/samples_5_10_22.csv'

    parser = argparse.ArgumentParser(description='Predict appliance\
                                     given a trained neural network\
                                     for energy disaggregation -\
                                     network input = mains window.')
    parser.add_argument('--appliances',
                        type=str,
                        nargs='+',
                        default=default_appliances,
                        help='name(s) of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default=default_dataset_dir,
                        help='directory location of test data')
    parser.add_argument('--rt_preds_datadir',
                        type=str,
                        default=default_rt_preds_dataset_dir,
                        help='directory location of real-time prediction dataset')
    parser.add_argument('--panel',
                        type=str,
                        default=default_panel_location,
                        help='sub-panel location')
    parser.add_argument('--trained_model_dir',
                        type=str,
                        default=default_model_dir,
                        help='directory to the trained models')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        default=default_ckpt_dir,
                        help='directory name of model checkpoint')
    parser.add_argument('--save_results_dir',
                        type=str,
                        default=default_results_dir,
                        help='directory to save the predictions')
    parser.add_argument('--plot', action='store_true',
                        help='show predicted appliance and mains power plots')
    parser.add_argument('--show_rt_preds', action='store_true',
                        help='show real-time predictions on plot')
    parser.add_argument('--threshold_rt_preds', action='store_true',
                        help='apply power on thresholds to real-time predictions on plot')                    
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='use part of the dataset for predictions')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1000,
                        help='mini-batch size')
    parser.set_defaults(plot=False)
    parser.set_defaults(show_rt_preds=False)
    parser.set_defaults(threshold_rt_preds=False)

    log(f'Machine name: {socket.gethostname()}')
    args = parser.parse_args()
    log('Arguments: ')
    log(args)

    log(f'Target appliance(s): {args.appliances}')

    # offset parameter from window length
    offset = int(0.5 * (WINDOW_LENGTH - 1.0))

    def load_dataset(file_name, crop=None) -> np.array:
        """Load input dataset file and return as np array.."""
        df = pd.read_csv(file_name, header=None, nrows=crop)

        return np.array(df, dtype=np.float32)

    test_set_x = load_dataset(os.path.join(
        args.datadir, f'{args.panel}.csv'), args.crop)
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
        # Apply on-power thresholds.
        threshold = params_appliance[appliance]['on_power_threshold']
        prediction[prediction <= threshold] = 0.0
        return prediction
    predictions = {
        appliance : prediction(
            appliance
        ) for appliance in args.appliances
    }

    log('aggregate_mean: ' + str(AGGREGATE_MEAN))
    log('aggregate_std: ' + str(AGGREGATE_STD))
    aggregate = test_set_x.flatten() * AGGREGATE_STD + AGGREGATE_MEAN

    # Calculate metrics. 
    # get_Epd returns a relative metric between two powers, so zero out one.
    target = np.zeros_like(aggregate)
    aggregate_epd = get_Epd(target, aggregate, SAMPLE_PERIOD)
    log(f'Aggregate energy: {aggregate_epd/1000:.3f} kWh per day')
    for appliance in args.appliances:
        epd = get_Epd(target, predictions[appliance], SAMPLE_PERIOD)
        log(f'{appliance} energy: {epd/1000:.3f} kWh per day')

    save_path = os.path.join(args.save_results_dir, args.panel)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Find max value in predictions for setting plot limits.
    max_pred = np.ceil(np.max(list(predictions.values())))

    if args.show_rt_preds:
        # Load real-time prediction dataset.
        df = pd.read_csv(args.rt_preds_datadir, usecols=default_appliances)
        df = df.fillna(0) # convert NaN's into zero's
        # Define real-time predictions columns to appliance names.
        # Select appliance prediction column then adjust for output timing.
        # Adjustment is simply moving samples earlier in time by
        # a WINDOW_LENGTH since the real-time code places the prediction
        # at the end of a window of samples.
        rt_preds_to_appliances = {
            appliance: np.array(
                df[[appliance]][WINDOW_LENGTH:], dtype=np.float32
            ) for appliance in args.appliances
        }

        # Apply power on thresholds.
        if args.threshold_rt_preds:
            for appliance in args.appliances:
                threshold = params_appliance[appliance]['on_power_threshold']
                rt_preds_to_appliances[appliance][rt_preds_to_appliances[appliance] <= threshold] = 0.0

    # Save and perhaps show powers in a single row of subplots.
    nrows = len(args.appliances) + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=1, constrained_layout=True)
    ax[0].set_ylabel('Watts')
    ax[0].set_title('aggregate')
    ax[0].plot(aggregate[offset:-offset], color='tab:orange', linewidth=1.8)
    row = 1
    for appliance in args.appliances:
        ax[row].set_ylabel('Watts')
        ax[row].set_title(appliance)
        ax[row].set_ylim(0, max_pred)
        ax[row].plot(
            predictions[appliance], color='tab:red', 
            linewidth=1.5, label='prediction'
        )
        if args.show_rt_preds:
            ax[row].plot(
                rt_preds_to_appliances[appliance], color='tab:green',
                linewidth=1.5, label='real-time prediction'
            )
            ax[row].legend(loc='upper right')
        row+=1
    fig.suptitle(f'Prediction results for {args.panel}',
        fontsize=16, fontweight='bold')
    plot_savepath = os.path.join(save_path, f'{args.panel}.png')
    plt.savefig(fname=plot_savepath)
    if args.plot:
        plt.show()
    plt.close()